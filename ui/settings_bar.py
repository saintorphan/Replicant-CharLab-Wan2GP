"""Shared generation settings bar — Replicant-owned, rendered once and used by
both Base Gen and Pose Gen (and reflected on every page). Spans both backends:
native (Flux/Z-Image/Qwen) and SD-family (SDXL/Pony/Illustrious)."""
from __future__ import annotations

import gradio as gr

from ..core import discovery

SAMPLERS = ["DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE", "Euler a", "Euler",
            "Heun", "DDIM", "UniPC", "LCM", "default"]
SCHEDULERS = ["", "Karras", "Exponential", "Normal", "SGM Uniform", "Simple"]


def build_settings_bar(model_choices=None, lora_choices=None):
    """Build the shared settings controls inside the current context. Returns a
    dict of components for the generator wiring."""
    c = {}
    with gr.Accordion("Generation settings", open=True, elem_classes="replicant-acc"):
        with gr.Row():
            c["model"] = gr.Dropdown(label="Model", choices=model_choices or [],
                                     scale=3)
            c["sampler"] = gr.Dropdown(label="Sampler", choices=SAMPLERS,
                                       value="DPM++ 2M", scale=1)
            c["scheduler"] = gr.Dropdown(label="Scheduler", choices=SCHEDULERS,
                                         value="Karras", scale=1)
        with gr.Row():
            c["steps"] = gr.Slider(1, 60, value=28, step=1, label="Steps")
            c["cfg_scale"] = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="CFG")
            c["clip_skip"] = gr.Slider(1, 4, value=2, step=1, label="Clip skip")
            c["seed"] = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
        with gr.Row():
            c["width"] = gr.Slider(256, 2048, value=832, step=64, label="Width")
            c["height"] = gr.Slider(256, 2048, value=1216, step=64, label="Height")
        with gr.Row():
            c["loras"] = gr.Dropdown(label="LoRAs", multiselect=True,
                                     choices=lora_choices or [], scale=3)
            c["lora_multipliers"] = gr.Textbox(label="Multipliers", placeholder="0.8, 1.0",
                                               scale=1)
    return c


def model_choices():
    """Convenience: SD-only choices for standalone use (native list comes from the
    plugin, which has the wgp globals)."""
    return discovery.build_model_choices()


def lora_choices():
    return [(m["name"], m["path"]) for m in discovery.discover_sdxl_loras()]
