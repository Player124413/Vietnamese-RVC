import os
import sys
import tqdm
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from main.library.predictors.Generator import Generator
from main.library.utils import load_audio, get_providers
from main.app.variables import config, logger, translations

class FeatureInput:
    def __init__(self, sample_rate=16000, hop_size=160, is_half=False, device=config.device):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.is_half = is_half

    def process_file(self, file_info, f0_method, hop_length, f0_onnx, f0_autotune, f0_autotune_strength):
        if not hasattr(self, "f0_gen"): self.f0_gen = Generator(self.fs, hop_length, self.f0_min, self.f0_max, self.is_half, self.device, get_providers(), f0_onnx)

        inp_path, opt_path1, opt_path2, file_inp = file_info
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            pitch, pitchf = self.f0_gen.calculator(config.x_pad, f0_method, load_audio(logger, file_inp, self.fs), 0, None, 0, f0_autotune, f0_autotune_strength, None, False)
            np.save(opt_path2, pitchf, allow_pickle=False)
            np.save(opt_path1, pitch, allow_pickle=False)
        except Exception as e:
            logger.info(f"{translations['extract_file_error']} {inp_path}: {e}")
            logger.debug(traceback.format_exc())

    def process_files(self, files, f0_method, hop_length, f0_onnx, device, is_half, threads, f0_autotune, f0_autotune_strength):
        self.device = device
        self.is_half = is_half

        def worker(file_info):
            self.process_file(file_info, f0_method, hop_length, f0_onnx, f0_autotune, f0_autotune_strength)

        with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                    pbar.update(1)