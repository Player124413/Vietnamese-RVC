import os
import sys
import json

sys.path.append(os.getcwd())

from main.app.variables import translations
from main.app.core.ui import gr_info, gr_warning, change_preset_choices

def load_presets(presets, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, formant_shifting, formant_qfrency, formant_timbre):
    if not presets: return gr_warning(translations["provide_file_settings"])

    with open(os.path.join("assets", "presets", presets)) as f:
        file = json.load(f)

    gr_info(translations["load_presets"].format(presets=presets))
    return file.get("cleaner", cleaner), file.get("autotune", autotune), file.get("pitch", pitch), file.get("clean_strength", clean_strength), file.get("index_strength", index_strength), file.get("resample_sr", resample_sr), file.get("filter_radius", filter_radius), file.get("volume_envelope", volume_envelope), file.get("protect", protect), file.get("split_audio", split_audio), file.get("f0_autotune_strength", f0_autotune_strength), file.get("formant_shifting", formant_shifting), file.get("formant_qfrency", formant_qfrency), file.get("formant_timbre", formant_timbre)

def save_presets(name, cleaner, autotune, pitch, clean_strength, index_strength, resample_sr, filter_radius, volume_envelope, protect, split_audio, f0_autotune_strength, cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox, formant_shifting, formant_qfrency, formant_timbre):  
    if not name: return gr_warning(translations["provide_filename_settings"])
    if not any([cleaner_chbox, autotune_chbox, pitch_chbox, index_strength_chbox, resample_sr_chbox, filter_radius_chbox, volume_envelope_chbox, protect_chbox, split_audio_chbox, formant_shifting_chbox]): return gr_warning(translations["choose1"])

    settings = {}

    for checkbox, data in [(cleaner_chbox, {"cleaner": cleaner, "clean_strength": clean_strength}), (autotune_chbox, {"autotune": autotune, "f0_autotune_strength": f0_autotune_strength}), (pitch_chbox, {"pitch": pitch}), (index_strength_chbox, {"index_strength": index_strength}), (resample_sr_chbox, {"resample_sr": resample_sr}), (filter_radius_chbox, {"filter_radius": filter_radius}), (volume_envelope_chbox, {"volume_envelope": volume_envelope}), (protect_chbox, {"protect": protect}), (split_audio_chbox, {"split_audio": split_audio}), (formant_shifting_chbox, {"formant_shifting": formant_shifting, "formant_qfrency": formant_qfrency, "formant_timbre": formant_timbre})]:
        if checkbox: settings.update(data)

    with open(os.path.join("assets", "presets", name + ".json"), "w") as f:
        json.dump(settings, f, indent=4)

    gr_info(translations["export_settings"])
    return change_preset_choices()