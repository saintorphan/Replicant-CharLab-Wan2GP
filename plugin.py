"""Replicant Character Lab — a Wan2GP plugin.

A guided 7-step wizard (Info → Prompt → Base → Face/Body → Poses → Save → Train)
for building a reusable character and its LoRA datasets, ported from
SupremeDiffusion's PySide6 character creator into Wan2GP's Gradio UI.

NOTE: not an official plugin. Distribute via the plugin-manager "add from GitHub
URL" flow; do not add to the bundled plugins.json without dbm's approval.
"""
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

try:  # GPU arbitration with the main Video Generator (see wan2gp-sample)
    from shared.utils.process_locks import (acquire_GPU_ressources,
                                            any_GPU_process_running,
                                            release_GPU_ressources)
    _HAVE_LOCKS = True
except Exception:  # pragma: no cover
    _HAVE_LOCKS = False

from .core import paths
from .ui import wizard
from .ui.styles import CSS

PLUGIN_ID = "ReplicantCharLab"
PLUGIN_NAME = "Replicant Character Lab"


class ReplicantCharLab(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.1.0"
        self.description = ("Character creator wizard: prompts, base image, poses, "
                            "LoRA datasets and training in a guided 7-step tab.")

    # -- lifecycle ----------------------------------------------------------
    def setup_ui(self):
        # First install / every boot: make sure the data dirs exist.
        try:
            paths.ensure_dirs()
        except Exception:
            traceback.print_exc()

        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")
        self.request_global("get_current_model_settings")
        # Native prompt enhancer (Qwen3.5 when enhancer_enabled in {3,4}).
        # exec_prompt_enhancer_engine self-manages the GPU lock + model load/unload.
        self.request_global("exec_prompt_enhancer_engine")
        self.request_global("get_state_model_type")
        self.request_global("get_model_def")

        self.add_tab(tab_id=PLUGIN_ID, label=PLUGIN_NAME,
                     component_constructor=self.create_ui)

    def on_tab_select(self, state: dict):
        """Block entering the lab while a Wan2GP generation is running: warn and
        bounce back to the Video Generator tab."""
        if _HAVE_LOCKS and any_GPU_process_running(state, PLUGIN_ID):
            gr.Warning("A generation is running — finish or stop it before using "
                       "the Character Lab.")
            try:
                return self.goto_video_tab(state)  # injected like in wan2gp-sample
            except Exception:
                return gr.update()
        return gr.update()

    # -- GPU arbitration helpers (used by step actions, wired later) --------
    def acquire_gpu(self, state):
        if not _HAVE_LOCKS:
            return True
        if any_GPU_process_running(state, PLUGIN_ID):
            gr.Error("Another process is using the GPU")
            return False
        acquire_GPU_ressources(state, PLUGIN_ID, PLUGIN_NAME, gr=gr)
        return True

    def release_gpu(self, state):
        if _HAVE_LOCKS:
            release_GPU_ressources(state, PLUGIN_ID)

    # -- UI -----------------------------------------------------------------
    def create_ui(self):
        gr.HTML(f"<style>{CSS}</style>")
        with gr.Column():
            ui = wizard.build_wizard()
        self._wire_enhancer(ui)
        # Outputs refreshed when the tab is (re)selected; bounce target is main_tabs.
        self.on_tab_outputs = [self.main_tabs] if hasattr(self, "main_tabs") else None
        self._ui = ui
        return ui

    def _wire_enhancer(self, ui):
        """Wire the Prompt step's Enhance buttons to Wan2GP's native enhancer."""
        prm = ui["components"]["prompt"]
        if not all(hasattr(self, a) for a in
                   ("exec_prompt_enhancer_engine", "get_state_model_type", "get_model_def", "state")):
            return  # globals not injected (older host) — leave buttons inert

        def _enhance(state, text, progress=gr.Progress()):
            if not (text and text.strip()):
                raise gr.Error("Enter or seed a prompt first.")
            model_type = self.get_state_model_type(state)
            model_def = self.get_model_def(model_type)
            out = self.exec_prompt_enhancer_engine(
                state, model_type, model_def,
                "T",            # text-only enhancement mode
                [text],         # original_prompts
                [None],         # image_start
                None,           # original_image_refs
                True,           # is_image
                False,          # audio_only
                -1,             # seed
                progress,
                -1,             # override_profile
                enhancer_kwargs={"image_prompt_type": "", "video_prompt_type": "",
                                 "audio_prompt_type": ""},
            )
            if out and out[0]:
                res = out[0]
                return res[0] if isinstance(res, (list, tuple)) else res
            return gr.update()

        prm["enhance_pos"].click(_enhance, inputs=[self.state, prm["positive_prompt"]],
                                 outputs=[prm["positive_prompt"]])
        prm["enhance_neg"].click(_enhance, inputs=[self.state, prm["negative_prompt"]],
                                 outputs=[prm["negative_prompt"]])


Plugin = ReplicantCharLab
