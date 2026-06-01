"""The seven wizard step panels.

Each ``build_*`` returns ``(group, components)`` where ``group`` is a ``gr.Group``
whose visibility the wizard toggles, and ``components`` is a dict of the
interactive widgets so the wizard can wire load/save/generation logic.

Generation-heavy actions (base gen, swaps, pose gen, training) are labelled
buttons; their GPU backends are wired post-move to Wan2GP's native pipelines and
the ported SupremeDiffusion helpers.
"""
from __future__ import annotations

import gradio as gr

from ..core import paths, poses

STEPS = [
    ("info", "① Info"),
    ("prompt", "② Prompt"),
    ("base", "③ Base Gen"),
    ("swap", "④ Face / Body"),
    ("poses", "⑤ Poses"),
    ("save", "⑥ Save"),
    ("train", "⑦ Train"),
]

STYLES = ["realism", "anime", "cartoon"]
# Generation settings live in the shared settings bar (ui/settings_bar.py), not
# per step — base/pose gen read from there.


def build_info(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ① Character Info")
        c = {}
        with gr.Row():
            c["load_existing"] = gr.Dropdown(label="Load existing character",
                                             choices=paths.list_characters(), scale=4)
            c["load_btn"] = gr.Button("Load", scale=0, min_width=90)
            c["refresh_btn"] = gr.Button("⟳", scale=0, min_width=44)
        c["name"] = gr.Textbox(label="Character name", placeholder="e.g. Nova")
        c["description"] = gr.Textbox(label="Description", lines=3,
            placeholder="a voluptuous woman with brown hair and glasses")
        c["style"] = gr.Radio(STYLES, value="realism", label="Style")
        gr.Markdown("<sub>Supplying a reference image skips **Base Gen** — the "
                    "reference becomes the base.</sub>")
        c["reference_image"] = gr.Image(label="Reference image (optional)",
                                        type="filepath", height=240)
        with gr.Accordion("LoRAs (optional)", open=False):
            c["selected_loras"] = gr.Dropdown(label="LoRAs", multiselect=True, choices=[])
            c["lora_multipliers"] = gr.Textbox(label="Multipliers", placeholder="0.8, 1.0")
            c["lora_trigger_words"] = gr.Textbox(label="Trigger words")
    return g, c


def build_prompt(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ② Prompts")
        gr.Markdown("<sub>Enhancement uses Wan2GP's abliterated Qwen3.5 enhancer.</sub>")
        c = {}
        with gr.Row():
            c["seed_prompt"] = gr.Button("Build seed from description")
            c["enhance_pos"] = gr.Button("✨ Enhance positive", variant="primary")
            c["enhance_neg"] = gr.Button("✨ Enhance negative")
        c["positive_prompt"] = gr.Textbox(label="Positive prompt", lines=4,
            placeholder="Seeded from the description + framing, then enhanced.")
        c["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=3)
    return g, c


def build_base(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ③ Base Generation")
        gr.Markdown("<sub>Generate full-body, front-facing candidates; pick one as "
                    "the canonical base for swaps and poses.</sub>")
        c = {}
        with gr.Row():
            c["count"] = gr.Slider(1, 8, value=4, step=1, label="Candidates")
            c["generate"] = gr.Button("Generate candidates", variant="primary", scale=2)
        with gr.Row():
            with gr.Column(scale=2):
                c["candidates"] = gr.Gallery(label="Candidates — click to select",
                                             columns=3, height=420)
            with gr.Column(scale=1):
                c["selected_base"] = gr.Image(label="Selected base", type="filepath",
                                              height=420)
    return g, c


def build_swap(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ④ Face / Body Swap")
        gr.Markdown("<sub>Optional — lock identity/body on the **base image**. Pose "
                    "generation always grabs the face from the finalized base.</sub>")
        c = {}
        with gr.Row():
            with gr.Column(scale=1):
                c["base_preview"] = gr.Image(label="Current base", type="filepath",
                                             height=300, interactive=False)
                c["result"] = gr.Image(label="Result (becomes the base)",
                                       type="filepath", height=300)
            with gr.Column(scale=1):
                gr.Markdown("**Face swap**")
                c["face_source"] = gr.Image(label="Face source", type="filepath", height=200)
                with gr.Row():
                    c["face_enhancer"] = gr.Radio(["", "gfpgan", "codeformer"], value="",
                                                  label="Enhancer")
                    c["face_enhancer_strength"] = gr.Slider(0.0, 1.0, value=0.5, label="Strength")
                c["face_blend_ratio"] = gr.Slider(0.0, 1.0, value=0.5, label="Enhancer blend")
                c["run_face"] = gr.Button("Apply face swap to base", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("**Body swap** (SDXL/Pony/Illustrious)")
                c["body_source"] = gr.Image(label="Body source", type="filepath", height=200)
                with gr.Row():
                    c["body_ip_scale"] = gr.Slider(0.0, 1.0, value=0.8, label="Identity")
                    c["body_denoise"] = gr.Slider(0.0, 1.0, value=0.75, label="Denoise")
                with gr.Row():
                    c["body_cfg"] = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="CFG")
                    c["body_cn_strength"] = gr.Slider(0.0, 1.0, value=0.7, label="ControlNet")
                c["run_body"] = gr.Button("Apply body swap to base", variant="primary")
    return g, c


def build_poses(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑤ Pose Variants")
        gr.Markdown(f"<sub>{len(poses.POSES)} predefined poses (full / medium / close, "
                    "varied angles). The base face is applied to each for identity "
                    "consistency, then you approve the keepers.</sub>")
        c = {}
        with gr.Row():
            c["ref_look_strength"] = gr.Slider(0.0, 1.0, value=0.7,
                label="Reference look strength (base → poses)")
            c["apply_body_to_poses"] = gr.Checkbox(value=True, label="Apply body swap to poses")
        c["generate"] = gr.Button("Generate poses", variant="primary")
        c["pose_gallery"] = gr.Gallery(label="Poses (approve to keep)", columns=4, height=340)
    return g, c


def build_save(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑥ Save Character")
        c = {}
        c["summary"] = gr.Markdown("_Fill in the earlier steps; the save summary "
                                   "appears here._")
        c["save"] = gr.Button("💾 Save character + build datasets", variant="primary")
        c["save_status"] = gr.Markdown()
    return g, c


def build_train(visible: bool):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑦ Train LoRA")
        gr.Markdown("<sub>Low-VRAM preset is auto-selected from your Wan2GP profile "
                    "(override below). The generation model is unloaded first to "
                    "free VRAM for training.</sub>")
        c = {}
        with gr.Row():
            c["dataset"] = gr.Radio(["video512", "highres", "full", "face"],
                                    value="video512", label="Dataset")
            c["base_model"] = gr.Dropdown(label="Base model", choices=[])
        with gr.Row():
            c["low_vram"] = gr.Radio(["auto (from profile)", "force low-VRAM", "full precision"],
                                     value="auto (from profile)", label="VRAM mode")
            c["epochs"] = gr.Slider(1, 50, value=16, step=1, label="Epochs")
        c["train"] = gr.Button("Start training", variant="primary")
        c["train_log"] = gr.Textbox(label="Training log", lines=8)
    return g, c


BUILDERS = [build_info, build_prompt, build_base, build_swap,
            build_poses, build_save, build_train]
