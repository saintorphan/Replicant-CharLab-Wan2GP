"""The seven wizard step panels.

Each ``build_*`` returns ``(group, components)`` where ``group`` is a ``gr.Group``
whose visibility the wizard toggles, and ``components`` is a dict of the
interactive widgets so the wizard can wire load/save/generation logic.

Generation-heavy actions (base gen, swaps, pose gen, training) are labelled
buttons; their GPU backends are wired post-move to Wan2GP's native pipelines and
the ported SupremeDiffusion helpers.
"""
from __future__ import annotations

import os

import gradio as gr

from ..core import paths, poses

STEPS = [
    ("setup", "① Setup"),
    ("base", "② Baseline"),
    ("swap", "③ Human Clone"),
    ("inpaint", "④ Touch Up"),
    ("poses", "⑤ Replicate"),
    ("train", "⑥ Train"),
]

STYLES = ["realism", "anime", "cartoon"]
# Generation settings live in the shared settings bar (ui/settings_bar.py), not
# per step — base/pose gen read from there.


def _init_img(init, key):
    """Restored image path for a component constructor (None if missing/gone)."""
    import os
    v = (init or {}).get(key)
    return v if (isinstance(v, str) and os.path.isfile(v)) else None


def _init_gallery(init, key):
    import os
    vs = (init or {}).get(key) or []
    out = [v for v in vs if isinstance(v, str) and os.path.isfile(v)]
    return out or None


def build_setup(visible: bool, init=None):
    """Combined Info + Prompt page."""
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ① Setup")
        c = {}
        with gr.Row():
            with gr.Column(scale=1):
                # Load/Save/Clear live in the header now (session actions).
                c["name"] = gr.Textbox(label="Character name", placeholder="e.g. Nova")
                c["description"] = gr.Textbox(label="Description", lines=3,
                    placeholder="a voluptuous woman with brown hair and glasses")
                c["style"] = gr.Radio(STYLES, value="realism", label="Style")
                gr.Markdown("**Prompts**  <sub>(enhancement uses the abliterated Qwen3.5 enhancer)</sub>")
                with gr.Row():
                    c["seed_prompt"] = gr.Button("Build seed from description")
                    c["enhance_pos"] = gr.Button("✨ Enhance positive", variant="primary")
                    c["enhance_neg"] = gr.Button("✨ Enhance negative")
                c["positive_prompt"] = gr.Textbox(label="Positive prompt", lines=4)
                c["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=3)
            with gr.Column(scale=1):
                c["reference_image"] = gr.Image(label="Reference image (optional — becomes the base)",
                                                type="filepath", height=760,
                                                show_fullscreen_button=True,
                                                value=_init_img(init, "setup.reference_image"))
        gr.Markdown("<sub>All fields autosave; restored next launch. LoRAs are picked in "
                    "the Generation settings bar.</sub>")
    return g, c


def build_base(visible: bool, init=None):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ② Baseline")
        c = {}
        with gr.Row():
            with gr.Column(scale=1):
                c["pos"] = gr.Textbox(label="Positive prompt (carried from Setup, editable)",
                                      lines=6, value=(init or {}).get("setup.positive_prompt", ""))
                c["neg"] = gr.Textbox(label="Negative prompt (carried from Setup, editable)",
                                      lines=4, value=(init or {}).get("setup.negative_prompt", ""))
            with gr.Column(scale=1):
                gr.Markdown("**Generate (txt2img)** — fresh candidates from the prompt.")
                c["count"] = gr.Slider(1, 8, value=4, step=1, label="Candidates")
                c["generate"] = gr.Button("Generate candidates (txt2img)", variant="primary")
                gr.Markdown("**Reimagine (img2img)** — re-render the reference (SD models). "
                            "Skip both and the reference passes through as the base.")
                c["denoise"] = gr.Slider(0.2, 1.0, value=0.6, step=0.05,
                                         label="Reimagine denoise")
                c["reimagine"] = gr.Button("Reimagine reference (img2img)")
        with gr.Row():
            with gr.Column(scale=2):
                # Tall enough for two full rows of portrait candidates (3 cols);
                # scrolls when there are more.
                c["candidates"] = gr.Gallery(label="Candidates — click to select one",
                                             columns=3, rows=2, height=720,
                                             object_fit="contain", show_fullscreen_button=True,
                                             value=_init_gallery(init, "base.candidates"))
                c["use_as_base"] = gr.Button("⬇ Use selected candidate as Base",
                                             variant="primary")
            with gr.Column(scale=1):
                c["selected_base"] = gr.Image(label="Base (changes only via the buttons)",
                                              type="filepath", height=560,
                                              interactive=False, show_fullscreen_button=True,
                                              value=_init_img(init, "base.selected_base"))
                c["revert_ref"] = gr.Button("↩ Revert to Reference",
                                            interactive=bool(_init_img(init, "setup.reference_image")))
                c["ref_avatar"] = gr.Image(label="Reference", type="filepath", height=200,
                                           interactive=False, show_fullscreen_button=True,
                                           value=_init_img(init, "setup.reference_image"))
        c["picked"] = gr.State(None)  # clicked candidate path (no base change until a button)
    return g, c


def build_swap(visible: bool, init=None):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ③ Human Clone")
        gr.Markdown("*This step is completely optional.*")
        c = {}
        with gr.Row():
            with gr.Column(scale=1):
                c["base_preview"] = gr.Image(label="Current base", type="filepath",
                                             height=560, interactive=False,
                                             show_fullscreen_button=True,
                                             value=_init_img(init, "base.selected_base"))
                c["ab_btn"] = gr.Button("🔍 A/B compare (full screen + zoom)",
                                        scale=0, min_width=200)
                c["result"] = gr.Image(label="Swap result (preview — Accept to make it the base)",
                                       type="filepath", height=560, interactive=False,
                                       show_fullscreen_button=True)
            with gr.Column(scale=1):
                gr.Markdown("### Face swap")
                c["face_source"] = gr.Image(label="Face source", type="filepath", height=320,
                                            show_fullscreen_button=True,
                                            value=_init_img(init, "swap.face_source"))
                with gr.Row():
                    c["face_enhancer"] = gr.Radio(["", "gfpgan", "codeformer"],
                                                  value="", label="Enhancer")
                    c["face_enhancer_strength"] = gr.Slider(0.0, 1.0, value=0.5, label="Strength")
                c["face_blend_ratio"] = gr.Slider(0.0, 1.0, value=0.5, label="Enhancer blend")
                c["face_adetailer"] = gr.Checkbox(value=False,
                    label="ADetailer (face detail pass — needs SDXL/Pony/Illustrious model)")
                with gr.Row(visible=False) as face_adet_row:  # toggled by the checkbox
                    c["face_adet_pos"] = gr.Textbox(label="ADetailer positive", lines=1, scale=1)
                    c["face_adet_neg"] = gr.Textbox(label="ADetailer negative", lines=1, scale=1)
                c["face_adet_row"] = face_adet_row
                with gr.Row():
                    c["run_face"] = gr.Button("Run face swap", variant="primary")
                    c["retry_face"] = gr.Button("↻ Retry", interactive=False)
                    c["accept_face"] = gr.Button("✓ Accept → base", variant="primary", interactive=False)
                gr.Markdown("### Body swap  <sub>(SDXL/Pony/Illustrious)</sub>")
                c["body_source"] = gr.Image(label="Body source", type="filepath", height=320,
                                            show_fullscreen_button=True,
                                            value=_init_img(init, "swap.body_source"))
                with gr.Row():
                    c["body_ip_scale"] = gr.Slider(0.0, 1.0, value=0.8, label="Identity")
                    c["body_denoise"] = gr.Slider(0.0, 1.0, value=0.75, label="Denoise")
                with gr.Row():
                    c["body_cfg"] = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="CFG")
                    c["body_cn_strength"] = gr.Slider(0.0, 1.0, value=0.7, label="ControlNet")
                c["adetailer"] = gr.Checkbox(value=True, label="ADetailer (face restore on body-swap result)")
                with gr.Row(visible=True) as body_adet_row:  # toggled by the checkbox
                    c["body_adet_pos"] = gr.Textbox(label="ADetailer positive", lines=1, scale=1)
                    c["body_adet_neg"] = gr.Textbox(label="ADetailer negative", lines=1, scale=1)
                c["body_adet_row"] = body_adet_row
                with gr.Row():
                    c["run_body"] = gr.Button("Run body swap", variant="primary")
                    c["retry_body"] = gr.Button("↻ Retry", interactive=False)
                    c["accept_body"] = gr.Button("✓ Accept → base", variant="primary", interactive=False)
                # Shown only while a body swap is running; cancels it mid-generation.
                c["abort_body"] = gr.Button("⛔ Abort body swap", variant="stop", visible=False)
        # A/B comparison overlay — base vs swapped, side by side, each zoomable full screen.
        with gr.Row(visible=False) as ab_row:
            with gr.Column():
                gr.Markdown("### A — Base")
                c["ab_base"] = gr.Image(type="filepath", height=640, interactive=False,
                                        show_label=False, show_fullscreen_button=True)
            with gr.Column():
                gr.Markdown("### B — Swapped")
                c["ab_result"] = gr.Image(type="filepath", height=640, interactive=False,
                                          show_label=False, show_fullscreen_button=True)
        with gr.Row(visible=False) as ab_close_row:
            c["ab_close"] = gr.Button("Close compare")
        c["ab_row"] = ab_row
        c["ab_close_row"] = ab_close_row
        # "idle" or "review" — lets the Run button double as Discard during review.
        c["face_mode"] = gr.State("idle")
        c["body_mode"] = gr.State("idle")
    return g, c


def build_inpaint(visible: bool, init=None, lora_choices=None):
    base_img = _init_img(init, "base.selected_base")
    _i = init or {}
    # Cohesion prompts default to the main Setup prompts, but a persisted cohesion
    # value (key present, even if cleared to "") takes precedence.
    cohesion_pos = _i.get("inpaint.cohesion_prompt", _i.get("setup.positive_prompt", ""))
    cohesion_neg = _i.get("inpaint.cohesion_neg", _i.get("setup.negative_prompt", ""))
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ④ Touch Up")
        gr.Markdown("*This step is completely optional.*")
        c = {}
        with gr.Tabs() as touchup_tabs:
          # --- Inpaint sub-tab: paint a mask on the base, prompt-driven inpaint --
          with gr.Tab("Inpaint", id="inpaint"):
            with gr.Row():
                with gr.Column(scale=1):
                    c["editor"] = gr.ImageEditor(label="Paint the area to fix", type="numpy",
                                                 height=720, layers=False, eraser=True,
                                                 transforms=[], value=base_img,
                                                 brush=gr.Brush(colors=["#ffffff"],
                                                                color_mode="fixed"))
                with gr.Column(scale=1, elem_id="replicant-inpaint-opts"):
                    c["load_base"] = gr.Button("⬆ Reload current base into canvas")
                    c["inpaint_prompt"] = gr.Textbox(label="Positive prompt", lines=3)
                    c["inpaint_neg"] = gr.Textbox(label="Negative prompt", lines=2)
                    c["inpaint_denoise"] = gr.Slider(0.2, 1.0, value=0.75, step=0.05,
                                                     label="Denoise")
                    c["inpaint_count"] = gr.Slider(1, 8, value=1, step=1,
                                                   label="Results")
                    c["inpaint_mask_blur"] = gr.Slider(0, 64, value=4, step=1,
                                                       label="Mask blur (px)")
                    c["inpaint_fill"] = gr.Radio(
                        ["fill", "original", "latent noise", "latent nothing"],
                        value="original", label="Masked content")
                    c["inpaint_full_res"] = gr.Checkbox(value=False,
                        label="Inpaint at full resolution (masked region only)")
                    c["inpaint_padding"] = gr.Slider(0, 256, value=32, step=4,
                        label="Full-res padding (px)")
                    gr.Markdown("<sub>Adjust additional settings in **Generation Settings** "
                                "above. Size is locked to portrait.</sub>")
                    with gr.Accordion("LoRAs", open=False):
                        c["inpaint_loras"] = gr.Dropdown(label="LoRAs", multiselect=True,
                                                         choices=lora_choices or [])
                        c["inpaint_lora_mult"] = gr.Textbox(label="Multipliers",
                                                            placeholder="0.8, 1.0")
            # Actions row — between the canvas/settings and the results strip.
            with gr.Row():
                c["run_inpaint"] = gr.Button("Run inpaint", variant="primary")
                c["use_inpaint"] = gr.Button("✓ Use as Base", variant="primary")
                c["reuse_inpaint"] = gr.Button("↻ Send to Inpaint")
                c["send_to_cohesion"] = gr.Button("→ Send to Cohesion")
                c["revert_inpaint"] = gr.Button("↩ Revert")
            # Output viewer: scrolls horizontally, tall enough for a full portrait.
            c["inpaint_gallery"] = gr.Gallery(label="Results", height=640, columns=20,
                rows=1, object_fit="contain", show_fullscreen_button=True,
                elem_id="replicant-inpaint-out")
          # --- Cohesion sub-tab: gentle img2img normalize pass ----------------
          with gr.Tab("Cohesion", id="cohesion"):
            with gr.Row():
                with gr.Column(scale=1):
                    c["cohesion_src"] = gr.Image(label="Source (current base)",
                                                 type="filepath", height=420,
                                                 interactive=False,
                                                 show_fullscreen_button=True, value=base_img)
                    c["cohesion_prompt"] = gr.Textbox(label="Positive prompt", lines=2,
                                                      value=cohesion_pos)
                    c["cohesion_neg"] = gr.Textbox(label="Negative prompt", lines=1,
                                                   value=cohesion_neg)
                    with gr.Row():
                        c["cohesion_enhance_pos"] = gr.Button("✨ Enhance positive")
                        c["cohesion_enhance_neg"] = gr.Button("✨ Enhance negative")
                    c["cohesion_cfg"] = gr.Slider(0.15, 0.30, value=0.22, step=0.01,
                                                  label="CFG (override)")
                    c["cohesion_steps"] = gr.Slider(5, 15, value=10, step=1,
                                                    label="Steps (override)")
                    c["cohesion_focus"] = gr.Textbox(label="Focus", lines=1,
                        placeholder="e.g. white background, even lighting")
                    c["normalize_btn"] = gr.Button("Normalize", variant="primary")
                with gr.Column(scale=1):
                    c["cohesion_gallery"] = gr.Gallery(label="Normalized (max 4) — click "
                        "to select, then Use as Base", columns=2, rows=2, height=560,
                        object_fit="contain", show_fullscreen_button=True)
                    c["use_cohesion"] = gr.Button("✓ Use selected as Base", variant="primary")
                    c["reuse_cohesion"] = gr.Button("↻ Send to Cohesion")
                    c["send_to_inpaint"] = gr.Button("→ Send to Inpaint")
        c["touchup_tabs"] = touchup_tabs
        c["inpaint_picked"] = gr.State(None)
        c["inpaint_prev_base"] = gr.State(None)  # for Revert
        c["cohesion_picked"] = gr.State(None)
    return g, c


def build_poses(visible: bool, init=None):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑤ Replicate")
        n = len(poses.POSES)
        gr.Markdown(f"<sub>{n} predefined poses (full / medium / close, varied angles). "
                    "Generate, then set each pose's dropdown and **Re-run poses**.</sub>")
        c = {}
        face_ref = bool(_init_img(init, "swap.face_source"))
        body_ref = bool(_init_img(init, "swap.body_source"))
        with gr.Row():
            c["ref_look_strength"] = gr.Slider(0.0, 1.0, value=0.7,
                label="Reference look strength (base → poses)")
            c["face_mode"] = gr.Radio(
                ["None", "Use Base"] + (["Use Reference"] if face_ref else []),
                value="Use Base", label="Face swap")
            c["body_mode"] = gr.Radio(
                ["None", "Use Base"] + (["Use Reference"] if body_ref else []),
                value="None", label="Body double")
        # ADetailer for poses — this tab's own settings (face uses the face model,
        # body uses the person model). Independent of the Human Clone tab.
        with gr.Row():
            c["pose_face_adet"] = gr.Checkbox(value=False, label="ADetailer (face)", scale=0,
                                              min_width=150)
            c["pose_face_adet_pos"] = gr.Textbox(show_label=False, scale=2, lines=1,
                                                 placeholder="face detail — positive")
            c["pose_face_adet_neg"] = gr.Textbox(show_label=False, scale=2, lines=1,
                                                 placeholder="face detail — negative")
        with gr.Row():
            c["pose_body_adet"] = gr.Checkbox(value=False, label="ADetailer (body)", scale=0,
                                              min_width=150)
            c["pose_body_adet_pos"] = gr.Textbox(show_label=False, scale=2, lines=1,
                                                 placeholder="body detail — positive")
            c["pose_body_adet_neg"] = gr.Textbox(show_label=False, scale=2, lines=1,
                                                 placeholder="body detail — negative")
        with gr.Row():
            c["generate"] = gr.Button("Generate poses", variant="primary")
            c["rerun"] = gr.Button("↻ Re-run poses (apply dropdowns)", variant="primary")
            c["abort_all"] = gr.Button("⛔ Abort all", variant="stop", scale=0,
                                       min_width=120)
        gr.Markdown("<sub>Per pose: **Approve** keeps it · **Sharpen (no upscale)** = crisp "
                    "the whole image (no model) · **Cohesion (img2img)** = gentle low-CFG "
                    "cleanup · **Re-Roll (img2img)** = heavier re-roll · **Regenerate "
                    "(txt2img)** = fresh image. **Color match** (only on Cohesion/Re-Roll) "
                    "retones the body to the base for skin-tone consistency.</sub>")
        # Fixed grid: one (image + dropdown + color-match) slot per pose.
        # Per-pose dropdown + color selections persist across reloads (saved to
        # wizard_state by _wire_persistence, restored here from init).
        saved = (init or {}).get("poses.pose_gallery") or []
        _ch = (init or {}).get("poses.choices") or []
        _co = (init or {}).get("poses.colors") or []
        _DD = ["Approve", "Sharpen (no upscale)", "Cohesion (img2img)",
               "Re-Roll (img2img)", "Regenerate (txt2img)"]
        c["pose_imgs"], c["pose_choices"], c["pose_color"] = [], [], []
        c["pose_abort"], c["pose_undo"] = [], []
        # Holds the pre-re-run image for each slot so the ↩ button can revert one
        # pose to exactly what it was before the last Re-run.
        c["pose_prev"] = gr.State(list(saved))
        for r in range(0, n, 4):
            with gr.Row():
                for idx in range(r, min(r + 4, n)):
                    with gr.Column(scale=1, min_width=180):
                        img = gr.Image(type="filepath", height=300, interactive=False,
                                       show_label=False, show_fullscreen_button=True,
                                       value=(saved[idx] if idx < len(saved) and
                                              os.path.isfile(str(saved[idx])) else None))
                        with gr.Row():
                            dd = gr.Dropdown(_DD, container=False, show_label=False,
                                             scale=5,
                                             value=(_ch[idx] if idx < len(_ch)
                                                    and _ch[idx] in _DD
                                                    else "Re-Roll (img2img)"))
                            un = gr.Button("↩", scale=0, min_width=40,
                                           elem_classes="replicant-pose-undo")
                            ab = gr.Button("⛔", variant="stop", scale=0, min_width=40,
                                           elem_classes="replicant-pose-abort")
                        cm = gr.Checkbox(label="Color match", container=False,
                                         value=(bool(_co[idx]) if idx < len(_co)
                                                else False))
                        c["pose_imgs"].append(img)
                        c["pose_choices"].append(dd)
                        c["pose_color"].append(cm)
                        c["pose_undo"].append(un)
                        c["pose_abort"].append(ab)
                        c["pose_abort"].append(ab)
    return g, c


def build_train(visible: bool, init=None):
    with gr.Group(visible=visible, elem_classes="replicant-step") as g:
        gr.Markdown("### ⑥ Datasets & Training")
        gr.Markdown("<sub>**Save Character** (header) writes the character + images to its "
                    "own folder. Then **Build LoRA datasets** here, and train — the low-VRAM "
                    "preset is auto-selected from your Wan2GP profile (override below) and "
                    "the generation model is unloaded first to free VRAM.</sub>")
        c = {}
        # Save summary (relocated here from the old Save page).
        c["summary"] = gr.Markdown("_Fill in the earlier steps, then **Save Character** "
                                   "in the header; the save summary appears here._")
        c["save_status"] = gr.Markdown()
        # Dataset creation tool.
        gr.Markdown("#### Datasets")
        c["build_datasets"] = gr.Button("🧱 Build LoRA datasets (from saved poses)",
                                        variant="primary")
        c["dataset_status"] = gr.Markdown()
        gr.Markdown("#### Training")
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


BUILDERS = [build_setup, build_base, build_swap, build_inpaint,
            build_poses, build_train]
