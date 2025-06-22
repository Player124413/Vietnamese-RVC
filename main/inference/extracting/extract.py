import os
import gc
import sys
import time
import tqdm
import torch
import logging
import argparse
import warnings
import traceback
import concurrent.futures

import numpy as np
import torch.multiprocessing as mp

from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.library import opencl
from main.inference.extracting.feature import FeatureInput
from main.inference.extracting.rms import RMSEnergyExtractor
from main.app.variables import config, logger, translations, configs
from main.inference.extracting.preparing_files import generate_config, generate_filelist
from main.library.utils import check_predictors, check_embedders, load_audio, load_embedders_model, get_providers, extract_features

warnings.filterwarnings("ignore")
for l in ["torch", "faiss", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="-")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--rms_extract", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def setup_paths(exp_dir, version = None, rms_extract = False):
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if rms_extract:
        out_path = os.path.join(exp_dir, "energy")
        os.makedirs(out_path, exist_ok=True)
        return wav_path, out_path

    if version:
        out_path = os.path.join(exp_dir, f"{version}_extracted")
        os.makedirs(out_path, exist_ok=True)
        return wav_path, out_path
    else:
        output_root1, output_root2 = os.path.join(exp_dir, "f0"), os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True); os.makedirs(output_root2, exist_ok=True)
        return wav_path, output_root1, output_root2

def run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, is_half, f0_autotune, f0_autotune_strength):
    num_processes = max(1, num_processes)
    input_root, *output_roots = setup_paths(exp_dir)
    output_root1, output_root2 = output_roots if len(output_roots) == 2 else (output_roots[0], None)

    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))
    num_processes = 1 if config.device.startswith("ocl") and (f0_method in "crepe" or f0_method in "fcpe" or f0_method in "rmvpe") else num_processes
    paths = [(os.path.join(input_root, name), os.path.join(output_root1, name) if output_root1 else None, os.path.join(output_root2, name) if output_root2 else None, os.path.join(input_root, name)) for name in sorted(os.listdir(input_root)) if "spec" not in name]

    start_time = time.time()
    feature_input = FeatureInput()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(feature_input.process_files, paths[i::len(devices)], f0_method, hop_length, f0_onnx, devices[i], is_half, num_processes // len(devices), f0_autotune, f0_autotune_strength) for i in range(len(devices))])
    
    gc.collect()
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def process_file_embedding(files, embedder_model, embedders_mode, device, version, is_half, threads):
    model, embed_suffix = load_embedders_model(embedder_model, embedders_mode, providers=get_providers())
    if embed_suffix != ".onnx": model = model.to(device).to(torch.float16 if is_half else torch.float32).eval()
    threads = max(1, threads)

    def worker(file_info):
        try:
            file, out_path = file_info
            out_file_path = os.path.join(out_path, os.path.basename(file.replace("wav", "npy")))
            if os.path.exists(out_file_path): return
            feats = torch.from_numpy(load_audio(logger, file, 16000)).to(device).to(torch.float16 if is_half else torch.float32).view(1, -1)

            with torch.no_grad():
                if embed_suffix == ".pt":
                    logits = model.extract_features(**{"source": feats, "padding_mask": torch.BoolTensor(feats.shape).fill_(False).to(device), "output_layer": 9 if version == "v1" else 12})
                    feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
                elif embed_suffix == ".onnx": feats = extract_features(model, feats, version).to(device)
                elif embed_suffix == ".safetensors":
                    logits = model(feats)["last_hidden_state"]
                    feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
                else: raise ValueError(translations["option_not_valid"])

            feats = feats.squeeze(0).float().cpu().numpy()
            if not np.isnan(feats).any(): np.save(out_file_path, feats, allow_pickle=False)
            else: logger.warning(f"{file} {translations['NaN']}")
        except:
            logger.debug(traceback.format_exc())

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, is_half):
    wav_path, out_path = setup_paths(exp_dir, version)
    start_time = time.time()

    logger.info(translations["start_extract_hubert"])
    num_processes = 1 if config.device.startswith("ocl") and embedders_mode == "onnx" else num_processes
    paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([executor.submit(process_file_embedding, paths[i::len(devices)], embedder_model, embedders_mode, devices[i], version, is_half, num_processes // len(devices)) for i in range(len(devices))])
    
    gc.collect()
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def process_file_rms(files, device, threads):
    threads = max(1, threads)

    module = RMSEnergyExtractor(
        frame_length=2048, hop_length=160, center=True, pad_mode = "reflect"
    ).to(device).eval().float()

    def worker(file_info):
        try:
            file, out_path = file_info
            out_file_path = os.path.join(out_path, os.path.basename(file))

            if os.path.exists(out_file_path + ".npy"): return
            with torch.no_grad():
                feats = torch.from_numpy(load_audio(logger, file, 16000)).unsqueeze(0)
                feats = module(feats if device.startswith("ocl") else feats.to(device))
                
            np.save(out_file_path, feats.float().cpu().numpy(), allow_pickle=False)
        except:
            logger.debug(traceback.format_exc())

    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_rms_extraction(exp_dir, num_processes, devices, rms_extract):
    if rms_extract:
        wav_path, out_path = setup_paths(exp_dir, rms_extract=rms_extract)
        start_time = time.time()
        paths = sorted([(os.path.join(wav_path, file), out_path) for file in os.listdir(wav_path) if file.endswith(".wav")])

        start_time = time.time()
        logger.info(translations["rms_start_extract"].format(num_processes=num_processes))

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
            concurrent.futures.wait([executor.submit(process_file_rms, paths[i::len(devices)], devices[i], num_processes // len(devices)) for i in range(len(devices))])

        logger.info(translations["rms_success_extract"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def main():
    args = parse_arguments()

    f0_method, hop_length, num_processes, gpus, version, pitch_guidance, sample_rate, embedder_model, f0_onnx, embedders_mode, f0_autotune, f0_autotune_strength, rms_extract = args.f0_method, args.hop_length, args.cpu_cores, args.gpu, args.rvc_version, args.pitch_guidance, args.sample_rate, args.embedder_model, args.f0_onnx, args.embedders_mode, args.f0_autotune, args.f0_autotune_strength, args.rms_extract
    exp_dir = os.path.join(configs["logs_path"], args.model_name)

    devices = ["cpu"] if gpus == "-" else [(f"ocl:{idx}" if opencl.is_available() else f"cuda:{idx}") for idx in gpus.split("-")]
    check_predictors(f0_method, f0_onnx=f0_onnx); check_embedders(embedder_model, embedders_mode)

    log_data = {translations['modelname']: args.model_name, translations['export_process']: exp_dir, translations['f0_method']: f0_method, translations['pretrain_sr']: sample_rate, translations['cpu_core']: num_processes, "Gpu": gpus, "Hop length": hop_length, translations['training_version']: version, translations['extract_f0']: pitch_guidance, translations['hubert_model']: embedder_model, translations["f0_onnx_mode"]: f0_onnx, translations["embed_mode"]: embedders_mode, translations["train&energy"]: rms_extract}
    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    try:
        run_pitch_extraction(exp_dir, f0_method, hop_length, num_processes, devices, f0_onnx, config.is_half, f0_autotune, f0_autotune_strength)
        run_embedding_extraction(exp_dir, version, num_processes, devices, embedder_model, embedders_mode, config.is_half)
        run_rms_extraction(exp_dir, num_processes, devices, rms_extract)
        generate_config(version, sample_rate, exp_dir)
        generate_filelist(pitch_guidance, exp_dir, version, sample_rate, embedders_mode, rms_extract)
    except Exception as e:
        logger.error(f"{translations['extract_error']}: {e}")

    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")

if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    main()