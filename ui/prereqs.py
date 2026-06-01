"""Prerequisites panel: collapsible Directories + Models sections, shown above
the wizard steps. Self-contained -- wires directly to core.paths / core.models."""
from __future__ import annotations

import gradio as gr

from ..core import models, paths


def _dir_row(label: str, value: str):
    """One path line: editable textbox + a 📁 browse toggle revealing a
    FileExplorer rooted at '/'. Selecting a folder fills the textbox."""
    with gr.Row():
        tb = gr.Textbox(value=value, label=label, scale=8, max_lines=1)
        browse = gr.Button("📁", scale=0, min_width=44)
    explorer = gr.FileExplorer(root_dir="/", label=f"Pick a folder for {label}",
                               file_count="single", visible=False)
    browse.click(lambda v: gr.update(visible=not v),
                 inputs=[gr.State(False)], outputs=[explorer])

    def _pick(sel):
        import os
        if not sel:
            return gr.update(), gr.update(visible=False)
        path = sel[0] if isinstance(sel, list) else sel
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        return gr.update(value=folder), gr.update(visible=False)

    explorer.change(_pick, inputs=[explorer], outputs=[tb, explorer])
    return tb


def build_prereqs():
    """Build the Prerequisites accordion. Returns a dict of key components."""
    with gr.Accordion("Prerequisites", open=False, elem_classes="replicant-acc"):
        with gr.Accordion("Directories", open=False, elem_classes="replicant-acc"):
            gr.Markdown("Where saved characters, datasets and this extension's "
                        "downloaded models live. Click 📁 to browse.")
            chars_tb = _dir_row("Characters root", str(paths.characters_dir()))
            data_tb = _dir_row("Dataset root", str(paths.datasets_dir()))
            models_tb = _dir_row("Models dir (download target)", str(paths.models_dir()))
            sdxl_m_tb = _dir_row("SDXL Models (SDXL/Pony/Illustrious checkpoints)",
                                 str(paths.sdxl_models_dir()))
            sdxl_l_tb = _dir_row("SDXL LoRAs (SDXL-family LoRA path)",
                                 str(paths.sdxl_loras_dir()))
            with gr.Row():
                save_btn = gr.Button("Save directories", variant="primary")
                dir_status = gr.Markdown()

            def _save(c, d, m, sm, sl):
                paths.set_dirs(characters=c, datasets=d, models=m,
                               sdxl_models=sm, sdxl_loras=sl)
                return (str(paths.characters_dir()), str(paths.datasets_dir()),
                        str(paths.models_dir()), str(paths.sdxl_models_dir()),
                        str(paths.sdxl_loras_dir()), "✅ Saved.")

            save_btn.click(_save, inputs=[chars_tb, data_tb, models_tb, sdxl_m_tb, sdxl_l_tb],
                           outputs=[chars_tb, data_tb, models_tb, sdxl_m_tb, sdxl_l_tb, dir_status])

        with gr.Accordion("Models (not bundled with Wan2GP)", open=False, elem_classes="replicant-acc"):
            gr.Markdown("Models this extension needs. Downloads stash into the "
                        "Models dir above.")
            rows = {}
            for spec in models.REGISTRY:
                with gr.Row():
                    st = gr.Markdown(_model_line(spec))
                    dl = gr.Button("Download", scale=0, min_width=110,
                                   interactive=bool(spec.url))

                def _dl(key=spec.key, _spec=spec):
                    def _run(progress=gr.Progress()):
                        msg = models.download(key, progress=progress)
                        return _model_line(_spec) + f"\n\n_{msg}_"
                    return _run

                dl.click(_dl(), outputs=[st])
                rows[spec.key] = st

    return {"chars_tb": chars_tb, "data_tb": data_tb, "models_tb": models_tb,
            "model_rows": rows}


def _model_line(spec) -> str:
    present = "✅ present" if spec.is_present() else (
        "⬇️ downloadable" if spec.url else "⚠️ source not set")
    req = "**required**" if spec.required else "optional"
    return (f"**{spec.name}** — {req} — {present}  \n"
            f"<sub>{spec.purpose} `{spec.subpath}`"
            + (f" — {spec.note}" if spec.note else "") + "</sub>")
