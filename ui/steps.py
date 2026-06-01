"""The seven wizard step panels.

Each ``build_*`` returns ``(group, components)`` where ``group`` is a
``gr.Group`` whose visibility the wizard toggles, and ``components`` is a dict of
the interactive widgets so the plugin can wire generation/save logic later.

Generation-heavy actions (base gen, swaps, pose gen, training) are intentionally
left as labelled buttons with no backend yet -- they will be wired to Wan2GP's
native pipelines and the ported SupremeDiffusion helpers once the move is done.
"""
from __future__ import annotations

import gradio as gr

STEPS = [
    ("info", "① Info"),
    ("prompt", "② Prompt"),
    ("base", "③ Base Gen"),
    ("swap", "④ Face / Body"),
    ("poses", "⑤ Poses"),
    ("save", "⑥ Save"),
    ("train", "⑦ Train"),
]


def build_info(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ① Character Info")
        c = {}
        c["name"] = gr.Textbox(label="Character name", placeholder="e.g. Nova")
        c["description"] = gr.Textbox(label="Description", lines=3,
            placeholder="a voluptuous woman with brown hair and glasses")
        c["style"] = gr.Radio(["realism", "anime", "cartoon"], value="realism", label="Style")
        c["reference_image"] = gr.Image(label="Reference image (optional)", type="filepath", height=220)
        c["selected_loras"] = gr.Dropdown(label="LoRAs (optional)", multiselect=True, choices=[])
    return g, c


def build_prompt(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ② Prompts")
        c = {}
        with gr.Row():
            c["enhance_pos"] = gr.Button("Enhance positive (Qwen3.5)", variant="primary")
            c["enhance_neg"] = gr.Button("Enhance negative (Qwen3.5)")
        c["positive_prompt"] = gr.Textbox(label="Positive prompt", lines=4)
        c["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=3)
    return g, c


def build_base(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ③ Base Generation")
        c = {}
        c["generate"] = gr.Button("Generate 4 candidates", variant="primary")
        c["candidates"] = gr.Gallery(label="Candidates", columns=4, height=260)
        c["selected_base"] = gr.Image(label="Selected base", type="filepath", height=240)
    return g, c


def build_swap(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ④ Face / Body Swap (optional — applied to the base image)")
        gr.Markdown("Optionally lock the identity/body on the **base** here. Pose "
                    "generation always grabs the face from the finalized base.")
        c = {}
        with gr.Tab("Face swap"):
            c["face_source"] = gr.Image(label="Face source", type="filepath", height=200)
            c["face_enhancer"] = gr.Radio(["", "gfpgan", "codeformer"], value="", label="Enhancer")
            c["face_blend_ratio"] = gr.Slider(0.0, 1.0, value=0.5, label="Enhancer blend")
            c["run_face"] = gr.Button("Apply face swap to base", variant="primary")
        with gr.Tab("Body swap"):
            c["body_source"] = gr.Image(label="Body source", type="filepath", height=200)
            c["body_ip_scale"] = gr.Slider(0.0, 1.0, value=0.8, label="Identity strength")
            c["body_denoise"] = gr.Slider(0.0, 1.0, value=0.75, label="Denoise")
            c["run_body"] = gr.Button("Apply body swap to base", variant="primary")
        c["result"] = gr.Image(label="Result (becomes the base)", type="filepath", height=240)
    return g, c


def build_poses(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑤ Pose Variants")
        gr.Markdown("Generates pose variants and applies the base face to each "
                    "(mandatory) for identity consistency.")
        c = {}
        c["generate"] = gr.Button("Generate poses", variant="primary")
        c["pose_gallery"] = gr.Gallery(label="Poses (approve to keep)", columns=4, height=340)
    return g, c


def build_save(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑥ Save Character")
        c = {}
        c["summary"] = gr.Markdown("_Nothing to save yet._")
        c["save"] = gr.Button("Save character + build datasets", variant="primary")
        c["save_status"] = gr.Markdown()
    return g, c


def build_train(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑦ Train LoRA")
        c = {}
        c["dataset"] = gr.Radio(["video512", "highres", "full", "face"],
                                value="video512", label="Dataset")
        c["base_model"] = gr.Dropdown(label="Base model", choices=[])
        c["train"] = gr.Button("Start training", variant="primary")
        c["train_log"] = gr.Textbox(label="Training log", lines=8)
    return g, c


BUILDERS = [build_info, build_prompt, build_base, build_swap,
            build_poses, build_save, build_train]
