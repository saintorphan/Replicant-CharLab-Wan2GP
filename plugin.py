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
        # Outputs refreshed when the tab is (re)selected; bounce target is main_tabs.
        self.on_tab_outputs = [self.main_tabs] if hasattr(self, "main_tabs") else None
        self._ui = ui
        return ui


Plugin = ReplicantCharLab
