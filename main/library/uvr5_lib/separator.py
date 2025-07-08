import os
import sys
import time
import yaml
import torch
import codecs
import hashlib
import requests
import onnxruntime

from importlib import import_module

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.library import opencl
from main.tools.huggingface import HF_download_file
from main.app.variables import config, translations

class Separator:
    def __init__(self, logger, model_file_dir=config.configs["uvr5_path"], output_dir=None, output_format="wav", output_bitrate=None, normalization_threshold=0.9, sample_rate=44100, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}, demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}):
        self.logger = logger
        self.logger.info(translations["separator_info"].format(output_dir=output_dir, output_format=output_format))
        self.model_file_dir = model_file_dir
        self.output_dir = output_dir if output_dir is not None else now_dir
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_format = output_format if output_format is not None else "wav"
        self.output_bitrate = output_bitrate
        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1: raise ValueError
        self.sample_rate = int(sample_rate)
        self.arch_specific_params = {"MDX": mdx_params, "Demucs": demucs_params}
        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None
        self.onnx_execution_provider = None
        self.model_instance = None
        self.model_friendly_name = None
        self.setup_torch_device()

    def setup_torch_device(self):
        hardware_acceleration_enabled = False
        ort_providers = onnxruntime.get_available_providers()
        self.torch_device_cpu = torch.device("cpu")

        if torch.cuda.is_available():
            self.configure_cuda(ort_providers)
            hardware_acceleration_enabled = True
        elif opencl.is_available():
            self.configure_amd(ort_providers)
            hardware_acceleration_enabled = True
        elif torch.backends.mps.is_available():
            self.configure_mps(ort_providers)
            hardware_acceleration_enabled = True

        if not hardware_acceleration_enabled:
            self.logger.info(translations["running_in_cpu"])
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CPUExecutionProvider"]

    def configure_cuda(self, ort_providers):
        self.logger.info(translations["running_in_cuda"])
        self.torch_device = torch.device("cuda")

        if "CUDAExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CUDAExecutionProvider'))
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CUDAExecutionProvider'))

    def configure_amd(self, ort_providers):
        self.logger.info(translations["running_in_amd"])
        self.torch_device = torch.device("ocl")

        if "DmlExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='DmlExecutionProvider'))
            self.onnx_execution_provider = ["DmlExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='DmlExecutionProvider'))

    def configure_mps(self, ort_providers):
        self.logger.info(translations["set_torch_mps"])
        self.torch_device_mps = torch.device("mps")
        self.torch_device = self.torch_device_mps

        if "CoreMLExecutionProvider" in ort_providers:
            self.logger.info(translations["onnx_have"].format(have='CoreMLExecutionProvider'))
            self.onnx_execution_provider = ["CoreMLExecutionProvider"]
        else: self.logger.warning(translations["onnx_not_have"].format(have='CoreMLExecutionProvider'))

    def get_model_hash(self, model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            return hashlib.md5(open(model_path, "rb").read()).hexdigest()

    def download_file_if_not_exists(self, url, output_path):
        if os.path.isfile(output_path): return
        HF_download_file(url, output_path)

    def list_supported_model_files(self):
        response = requests.get(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/hie_zbqryf.wfba", "rot13"))
        response.raise_for_status()
        model_downloads_list = response.json()

        return {"MDX": {**model_downloads_list["mdx_download_list"], **model_downloads_list["mdx_download_vip_list"]}, "Demucs": {key: value for key, value in model_downloads_list["demucs_download_list"].items() if key.startswith("Demucs v4")}}
    
    def download_model_files(self, model_filename):
        model_path = os.path.join(self.model_file_dir, model_filename)
        supported_model_files_grouped = self.list_supported_model_files()
        yaml_config_filename = None

        for model_type, model_list in supported_model_files_grouped.items():
            for model_friendly_name, model_download_list in model_list.items():
                model_repo_url_prefix = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/hie5_zbqryf", "rot13")

                if isinstance(model_download_list, str) and model_download_list == model_filename:
                    self.model_friendly_name = model_friendly_name

                    try:
                        self.download_file_if_not_exists(f"{model_repo_url_prefix}/MDX/{model_filename}", model_path)
                    except RuntimeError:
                        self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{model_filename}", model_path)

                    return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename
                elif isinstance(model_download_list, dict):
                    this_model_matches_input_filename = False

                    for file_name, file_url in model_download_list.items():
                        if file_name == model_filename or file_url == model_filename: this_model_matches_input_filename = True

                    if this_model_matches_input_filename:
                        self.model_friendly_name = model_friendly_name

                        for config_key, config_value in model_download_list.items():
                            if config_value.startswith("http"): self.download_file_if_not_exists(config_value, os.path.join(self.model_file_dir, config_key))
                            elif config_key.endswith(".ckpt"):
                                self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{config_key}", os.path.join(self.model_file_dir, config_key))

                                if model_filename.endswith(".yaml"):
                                    model_filename = config_key
                                    model_path = os.path.join(self.model_file_dir, f"{model_filename}")

                                yaml_config_filename = config_value
                                yaml_config_filepath = os.path.join(self.model_file_dir, yaml_config_filename)

                                self.download_file_if_not_exists(f"{model_repo_url_prefix}/mdx_c_configs/{yaml_config_filename}", yaml_config_filepath)
                            else: self.download_file_if_not_exists(f"{model_repo_url_prefix}/Demucs/{config_value}", os.path.join(self.model_file_dir, config_value))

                        return model_filename, model_type, model_friendly_name, model_path, yaml_config_filename

        raise ValueError

    def load_model_data_from_yaml(self, yaml_config_filename):
        model_data_yaml_filepath = os.path.join(self.model_file_dir, yaml_config_filename) if not os.path.exists(yaml_config_filename) else yaml_config_filename
        model_data = yaml.load(open(model_data_yaml_filepath, encoding="utf-8"), Loader=yaml.FullLoader)

        if "roformer" in model_data_yaml_filepath: model_data["is_roformer"] = True
        return model_data

    def load_model_data_using_hash(self, model_path):
        model_hash = self.get_model_hash(model_path)
        mdx_model_data_path = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/enj/znva/wfba/zbqry_qngn.wfba", "rot13")
        response = requests.get(mdx_model_data_path)
        response.raise_for_status()
        mdx_model_data_object = response.json()

        if model_hash in mdx_model_data_object: model_data = mdx_model_data_object[model_hash]
        else: raise ValueError

        return model_data

    def load_model(self, model_filename):
        self.logger.info(translations["loading_model"].format(model_filename=model_filename))
        model_filename, model_type, model_friendly_name, model_path, yaml_config_filename = self.download_model_files(model_filename)

        if model_path.lower().endswith(".yaml"): yaml_config_filename = model_path

        common_params = {"logger": self.logger, "torch_device": self.torch_device, "torch_device_cpu": self.torch_device_cpu, "torch_device_mps": self.torch_device_mps, "onnx_execution_provider": self.onnx_execution_provider, "model_name": model_filename.split(".")[0], "model_path": model_path, "model_data": self.load_model_data_from_yaml(yaml_config_filename) if yaml_config_filename is not None else self.load_model_data_using_hash(model_path), "output_format": self.output_format, "output_bitrate": self.output_bitrate, "output_dir": self.output_dir, "normalization_threshold": self.normalization_threshold, "output_single_stem": None, "invert_using_spec": False, "sample_rate": self.sample_rate}
        separator_classes = {"MDX": "mdx_separator.MDXSeparator", "Demucs": "demucs_separator.DemucsSeparator"}

        if model_type not in self.arch_specific_params or model_type not in separator_classes: raise ValueError(translations["model_type_not_support"].format(model_type=model_type))

        module_name, class_name = separator_classes[model_type].split(".")
        separator_class = getattr(import_module(f"main.library.architectures.{module_name}"), class_name)
        self.model_instance = separator_class(common_config=common_params, arch_config=self.arch_specific_params[model_type])

    def separate(self, audio_file_path):
        self.logger.info(f"{translations['starting_separator']}: {audio_file_path}")
        separate_start_time = time.perf_counter()
        output_files = self.model_instance.separate(audio_file_path)

        self.model_instance.clear_gpu_cache()
        self.model_instance.clear_file_specific_paths()

        self.logger.debug(translations["separator_success_3"])
        self.logger.info(f"{translations['separator_duration']}: {time.strftime('%H:%M:%S', time.gmtime(int(time.perf_counter() - separate_start_time)))}")
        return output_files