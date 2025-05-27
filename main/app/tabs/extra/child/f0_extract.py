import os
import sys
import shutil

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.f0_extract import f0_extract
from main.app.core.ui import change_audios_choices
from main.app.variables import translations, paths_for_files, method_f0

def f0_extract_tab():
    with gr.Row():
        gr.Markdown(translations["f0_extractor_markdown_2"])
    with gr.Row():
        extractor_button = gr.Button(translations["extract_button"].replace("2. ", ""), variant="primary")
    with gr.Row():
        with gr.Column():
            upload_audio_file = gr.File(label=translations["drop_audio"], file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".alac", ".wma", ".aiff", ".webm", ".ac3"])
            audioplay = gr.Audio(show_download_button=True, interactive=False, label=translations["input_audio"])
        with gr.Column():
            with gr.Accordion(translations["f0_method"], open=False):
                with gr.Group():
                    onnx_f0_mode3 = gr.Checkbox(label=translations["f0_onnx_mode"], info=translations["f0_onnx_mode_info"], value=False, interactive=True)
                    f0_method_extract = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
            with gr.Accordion(translations["audio_path"], open=True):
                input_audio_path = gr.Dropdown(label=translations["audio_path"], value="", choices=paths_for_files, allow_custom_value=True, interactive=True)
                refesh_audio_button = gr.Button(translations["refesh"])
    with gr.Row():
        gr.Markdown("___")
    with gr.Row():
        file_output = gr.File(label="", file_types=[".txt"], interactive=False)
        image_output = gr.Image(label="", interactive=False, show_download_button=True)
    with gr.Row():
        upload_audio_file.upload(fn=lambda audio_in: shutil.move(audio_in.name, os.path.join("audios")), inputs=[upload_audio_file], outputs=[input_audio_path])
        input_audio_path.change(fn=lambda audio: audio if os.path.isfile(audio) else None, inputs=[input_audio_path], outputs=[audioplay])
        refesh_audio_button.click(fn=change_audios_choices, inputs=[input_audio_path], outputs=[input_audio_path])
    with gr.Row():
        extractor_button.click(
            fn=f0_extract,
            inputs=[
                input_audio_path,
                f0_method_extract,
                onnx_f0_mode3
            ],
            outputs=[file_output, image_output],
            api_name="f0_extract"
        )