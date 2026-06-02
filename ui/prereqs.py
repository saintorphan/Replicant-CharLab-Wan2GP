"""OrphanSuite panel: collapsible Character-Lab folders + shared models/folders +
the model-download list, shown above the wizard steps. Self-contained -- wires
directly to core.paths / core.models.

Shared model dirs (download target, SDXL checkpoints, SDXL LoRAs) live in the
cross-plugin ``.orphansuite.json`` so every saintorphan plugin (Image Suite,
Replicant CharLab, Reel2Reel) follows the same paths; the Character-Lab folders
(characters, datasets) stay plugin-specific."""
from __future__ import annotations

import gradio as gr

from ..core import models, paths, presets
from .settings_bar import SAMPLERS, SCHEDULERS


def _dir_row(label: str, value: str):
    """One path line: editable textbox + a 📁 browse toggle revealing a
    FileExplorer. The browse root follows paths.browse_root() (full FS locally,
    auto-confined under --listen/--share, overridable in OrphanSuite). Selecting a
    folder fills the textbox; the textbox stays editable so any path can be typed."""
    with gr.Row():
        tb = gr.Textbox(value=value, label=label, scale=8, max_lines=1)
        browse = gr.Button("📁", scale=0, min_width=44)
    explorer = gr.FileExplorer(root_dir=paths.browse_root(),
                               label=f"Pick a folder for {label}",
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
    """Build the OrphanSuite accordion. Returns a dict of key components."""
    with gr.Accordion("OrphanSuite", open=False, elem_classes="replicant-acc"):
        # -- Character-Lab folders (this plugin only) --
        with gr.Accordion("Character Lab folders", open=False,
                          elem_classes="replicant-acc"):
            gr.Markdown("Where this plugin's saved characters and training "
                        "datasets live. Click 📁 to browse.")
            chars_tb = _dir_row("Characters root", str(paths.characters_dir()))
            data_tb = _dir_row("Dataset root", str(paths.datasets_dir()))
            with gr.Row():
                save_lab_btn = gr.Button("Save Character Lab folders", variant="primary")
                lab_status = gr.Markdown()

            def _save_lab(c, d):
                paths.set_dirs(characters=c, datasets=d)
                return (str(paths.characters_dir()), str(paths.datasets_dir()),
                        "✅ Saved.")

            save_lab_btn.click(_save_lab, inputs=[chars_tb, data_tb],
                               outputs=[chars_tb, data_tb, lab_status])

        # -- Shared models & folders (cross-plugin via .orphansuite.json) --
        with gr.Accordion("Shared models & folders (OrphanSuite)", open=False,
                          elem_classes="replicant-acc"):
            gr.Markdown(
                "**Shared across all saintorphan plugins** (Image Suite, Replicant "
                "CharLab, Reel2Reel) via `.orphansuite.json` — set a folder here and "
                "every plugin follows. Point them anywhere you already keep models so "
                "nothing's duplicated.")
            models_tb = _dir_row("Models dir (download target — face / ADetailer / "
                                 "BiRefNet)", str(paths.models_dir()))
            sdxl_m_tb = _dir_row("SDXL Models (SDXL/Pony/Illustrious checkpoints)",
                                 str(paths.sdxl_models_dir()))
            sdxl_l_tb = _dir_row("SDXL LoRAs (SDXL-family LoRA path)",
                                 str(paths.sdxl_loras_dir()))
            with gr.Row():
                save_shared_btn = gr.Button("Save shared folders", variant="primary")
                shared_status = gr.Markdown()

            def _save_shared(m, sm, sl):
                paths.set_dirs(models=m, sdxl_models=sm, sdxl_loras=sl)
                return (str(paths.models_dir()), str(paths.sdxl_models_dir()),
                        str(paths.sdxl_loras_dir()), "✅ Saved.")

            save_shared_btn.click(
                _save_shared, inputs=[models_tb, sdxl_m_tb, sdxl_l_tb],
                outputs=[models_tb, sdxl_m_tb, sdxl_l_tb, shared_status])

            gr.Markdown(
                "**Link an existing folder** — symlink models you already keep "
                "(a1111 / Forge / anywhere on disk) into the shared area. Works with "
                "physical files *or* symlinks and never moves the originals.")
            with gr.Row():
                link_src = gr.Textbox(label="Folder to link from",
                                      placeholder="/path/to/your/models", scale=3)
                link_target = gr.Dropdown(
                    label="Into", value="sdxl_models",
                    # Targets resolve to the exact dir each loader scans (incl. the
                    # face/body/birefnet subdirs) — see paths.link_target_dir.
                    choices=[("SDXL checkpoints", "sdxl_models"),
                             ("SDXL LoRAs", "sdxl_loras"),
                             ("Face / swap weights", "face"),
                             ("ADetailer / person-seg (body)", "body"),
                             ("BiRefNet (body-swap seg)", "birefnet"),
                             ("InsightFace buffalo_l (face detect)", "buffalo_l")],
                    scale=2)
                link_btn = gr.Button("🔗 Link", scale=1)
            link_status = gr.Markdown()

            def _link(src, leaf):
                if not (src and src.strip()):
                    return "⚠️ Enter a folder to link from."
                try:
                    return "✅ " + paths.link_existing_into_shared(src.strip(), leaf)
                except Exception as e:
                    return f"⚠️ {e}"

            link_btn.click(_link, inputs=[link_src, link_target], outputs=[link_status])

            gr.Markdown(
                "**Folder-browser root** — what the 📁 pickers above can browse "
                "(shared across plugins). Blank = auto: the whole filesystem locally, "
                "but auto-confined to your home folder when the app runs with "
                "`--listen`/`--share` (you only need the browser for local setup). "
                "Set a path to force it (e.g. a models drive, or your home to lock it "
                "down). **Applies on app restart.**")
            with gr.Row():
                browse_root_tb = gr.Textbox(
                    label="Folder-browser root (blank = auto)",
                    value=paths.get_shared("fs_browse_root", ""),
                    placeholder="(auto) — or e.g. /mnt/data4  or  " + str(paths.lab_root().home()),
                    scale=4)
                browse_root_save = gr.Button("Save browser root", scale=1)
            browse_root_status = gr.Markdown()

            def _save_browse_root(v):
                paths.set_shared("fs_browse_root", (v or "").strip())
                eff = paths.browse_root()
                return f"✅ Saved. Browser root → `{eff}` (effective on restart)."

            browse_root_save.click(_save_browse_root, inputs=[browse_root_tb],
                                   outputs=[browse_root_status])

        # -- Default Generation Values (per family; shared via .orphansuite.json) --
        with gr.Accordion("Default Generation Values (per family)", open=False,
                          elem_classes="replicant-acc"):
            gr.Markdown(
                "Recommended cfg / steps / sampler / scheduler + portrait resolution "
                "that auto-fill **Generation settings** when you pick a model. Edit "
                "and **Save** to set your own defaults (shared across all saintorphan "
                "plugins via `.orphansuite.json`); **Reset** restores the factory "
                "values. For Flux / Z-Image / Qwen the model's own steps/CFG still "
                "take precedence unless you save an override here.")
            _f0 = presets.FAMILIES[0]
            gd_fam = gr.Dropdown(label="Model family", choices=presets.FAMILIES,
                                 value=_f0)
            _e0 = presets.effective(_f0)
            with gr.Row():
                gd_steps = gr.Slider(1, 60, value=_e0["steps"], step=1, label="Steps")
                gd_cfg = gr.Slider(1.0, 15.0, value=_e0["cfg"], step=0.5, label="CFG")
                gd_clip = gr.Slider(1, 4, value=_e0["clip_skip"], step=1,
                                    label="Clip skip")
            with gr.Row():
                gd_sampler = gr.Dropdown(SAMPLERS, value=_e0["sampler"], label="Sampler")
                gd_scheduler = gr.Dropdown(SCHEDULERS, value=_e0["scheduler"],
                                           label="Scheduler")
            with gr.Row():
                gd_width = gr.Slider(256, 2048, value=_e0["width"], step=64,
                                     label="Width")
                gd_height = gr.Slider(256, 2048, value=_e0["height"], step=64,
                                      label="Height")
            with gr.Row():
                gd_save = gr.Button("Save as my default", variant="primary")
                gd_reset = gr.Button("Reset to factory")
            gd_status = gr.Markdown(f"Showing **{_f0}** "
                                    + ("(your override)." if presets.has_override(_f0)
                                       else "(factory)."))

            _GD = [gd_steps, gd_cfg, gd_sampler, gd_scheduler, gd_clip,
                   gd_width, gd_height]

            def _gd_fields(e):
                return [gr.update(value=e[k]) for k in
                        ("steps", "cfg", "sampler", "scheduler", "clip_skip",
                         "width", "height")]

            def _gd_load(fam):
                tag = "(your override)" if presets.has_override(fam) else "(factory)"
                return _gd_fields(presets.effective(fam)) + [f"Showing **{fam}** {tag}."]

            gd_fam.change(_gd_load, inputs=[gd_fam], outputs=_GD + [gd_status])

            def _gd_save(fam, steps, cfg, sampler, scheduler, clip, w, h):
                presets.set_overrides(fam, {
                    "steps": int(steps), "cfg": float(cfg), "sampler": sampler,
                    "scheduler": scheduler, "clip_skip": int(clip),
                    "width": int(w), "height": int(h)})
                return f"✅ Saved **{fam}** defaults (applies on next model select)."

            gd_save.click(_gd_save, inputs=[gd_fam] + _GD, outputs=[gd_status])

            def _gd_reset(fam):
                presets.clear_overrides(fam)
                return _gd_fields(presets.effective(fam)) + [
                    f"↺ **{fam}** reset to factory."]

            gd_reset.click(_gd_reset, inputs=[gd_fam], outputs=_GD + [gd_status])

        # -- Models registry (download buttons; nothing auto-downloads) --
        with gr.Accordion("Models (not bundled with Wan2GP)", open=False,
                          elem_classes="replicant-acc"):
            gr.Markdown("Models this extension needs. Downloads stash into the "
                        "shared Models dir above.")
            rows = {}
            for spec in models.REGISTRY:
                with gr.Row():
                    st = gr.Markdown(_model_line(spec))
                    dl = gr.Button("Download", scale=0, min_width=110,
                                   interactive=spec.downloadable)

                def _dl(key=spec.key, _spec=spec):
                    def _run(progress=gr.Progress()):
                        msg = models.download(key, progress=progress)
                        return _model_line(_spec) + f"\n\n_{msg}_"
                    return _run

                dl.click(_dl(), outputs=[st])
                rows[spec.key] = st

    return {"chars_tb": chars_tb, "data_tb": data_tb, "models_tb": models_tb,
            "sdxl_m_tb": sdxl_m_tb, "sdxl_l_tb": sdxl_l_tb, "model_rows": rows}


def _model_line(spec) -> str:
    present = "✅ present" if spec.is_present() else (
        "⬇️ click Download" if spec.downloadable else "⚠️ source not set")
    req = "**required**" if spec.required else "optional"
    return (f"**{spec.name}** — {req} — {present}  \n"
            f"<sub>{spec.purpose} `{spec.display_path()}`"
            + (f" — {spec.note}" if spec.note else "") + "</sub>")
