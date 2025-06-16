import os
import sys
import shutil

from random import shuffle

sys.path.append(os.getcwd())

from main.app.core.ui import configs

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path): shutil.copy(os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), config_save_path)

def generate_filelist(pitch_guidance, model_path, rvc_version, sample_rate, embedders_mode = "fairseq", rms_extract = False):
    gt_wavs_dir, feature_dir = os.path.join(model_path, "sliced_audios"), os.path.join(model_path, f"{rvc_version}_extracted")
    f0_dir, f0nsf_dir, energy_dir = None, None, None

    if pitch_guidance: f0_dir, f0nsf_dir = os.path.join(model_path, "f0"), os.path.join(model_path, "f0_voiced")
    if rms_extract: energy_dir = os.path.join(model_path, "energy")

    gt_wavs_files, feature_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), set(name.split(".")[0] for name in os.listdir(feature_dir))
    names = gt_wavs_files & feature_files

    if pitch_guidance: names = names & set(name.split(".")[0] for name in os.listdir(f0_dir)) & set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    if rms_extract: names = names & set(name.split(".")[0] for name in os.listdir(energy_dir))
    
    options = []
    mute_base_path = os.path.join(configs["logs_path"], "mute")

    for name in names:
        if pitch_guidance:
            if rms_extract:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{energy_dir}/{name}.wav.npy|0"
            else:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
        else:
            if rms_extract:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{energy_dir}/{name}.wav.npy|0"
            else:
                option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|0"

        options.append(option)

    mute_audio_path, mute_feature_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), os.path.join(mute_base_path, f"{rvc_version}_extracted", f"mute{'_spin' if embedders_mode == 'spin' else ''}.npy")
    
    for _ in range(2):
        if pitch_guidance:
            if rms_extract:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0"
            else:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'f0', 'mute.wav.npy')}|{os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')}|0"
        else:
            if rms_extract:
                option = f"{mute_audio_path}|{mute_feature_path}|{os.path.join(mute_base_path, 'energy', 'mute.wav.npy')}|0"
            else:
                option = f"{mute_audio_path}|{mute_feature_path}|0"

        options.append(option)

    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))