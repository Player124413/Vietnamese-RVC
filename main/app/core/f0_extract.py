import os
import sys
import librosa

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from main.inference.extract import FeatureInput
from main.library.utils import check_predictors
from main.app.core.ui import gr_info, gr_warning
from main.app.variables import config, translations

def f0_extract(audio, f0_method, f0_onnx):
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2

    check_predictors(f0_method, f0_onnx)

    f0_path = os.path.join("assets", "f0", os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])

    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    y, sr = librosa.load(audio, sr=None)

    feats = FeatureInput(sample_rate=sr, is_half=config.is_half, device=config.device)
    feats.f0_max = 1600.0

    F_temp = np.array(feats.compute_f0(y.flatten(), f0_method, 160, f0_onnx), dtype=np.float32)
    F_temp[F_temp == 0] = np.nan

    f0 = 1200 * np.log2(F_temp / librosa.midi_to_hz(0))

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(f0):
            f.write(f"{i * sr / 160},{f0_value}\n")

    gr_info(translations["extract_done"])

    return [txt_path, image_path]