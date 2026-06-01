"""The wizard shell: logo banner, clickable step rail, Back/Next nav over seven
visibility-toggled step panels (Gradio 5.29 has no native stepper component)."""
from __future__ import annotations

import base64
import os
import traceback
from pathlib import Path

import gradio as gr

from ..core import character, datasets, paths, wizard_state
from .prereqs import build_prereqs
from .settings_bar import build_settings_bar
from .steps import BUILDERS, STEPS

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_N = len(STEPS)


def _logo_data_uri() -> str:
    png = _ASSETS / "replicant.png"
    try:
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _banner_html() -> str:
    uri = _logo_data_uri()
    if uri:
        return f'<div id="replicant-banner"><img src="{uri}" alt="Replicant Character Lab"/></div>'
    return '<div id="replicant-banner"><h2>Replicant · Character Lab</h2></div>'


def build_wizard(model_choices=None, lora_choices=None, init=None):
    """Build the wizard UI inside the current tab context.

    model_choices/lora_choices populate the shared settings bar (the plugin
    supplies them since it has the wgp globals for native models).

    Returns a dict with ``step``, ``groups``, ``rail``, ``nav``, ``components``,
    ``settings`` (shared gen-settings bar) and ``prereqs``."""
    gr.HTML(_banner_html())

    # Prerequisites (directories + models) ----------------------------------
    prereqs = build_prereqs()

    # Shared generation settings (Replicant-owned, used across pages) --------
    settings = build_settings_bar(model_choices, lora_choices)

    # Step rail -------------------------------------------------------------
    rail = []
    with gr.Row(elem_id="replicant-rail"):
        for i, (_key, title) in enumerate(STEPS):
            rail.append(gr.Button(title, variant="primary" if i == 0 else "secondary"))

    # Step panels -----------------------------------------------------------
    groups, comps = [], {}
    for i, builder in enumerate(BUILDERS):
        g, c = builder(visible=(i == 0))
        groups.append(g)
        comps[STEPS[i][0]] = c

    # Nav -------------------------------------------------------------------
    with gr.Row(elem_id="replicant-nav"):
        back_btn = gr.Button("◀ Back", interactive=False)
        next_btn = gr.Button("Next ▶", variant="primary")

    step = gr.State(0)
    poses_state = gr.State({"poses": [], "specs": []})  # filled by pose gen (step 5)
    has_ref = gr.State(False)  # whether a reference image is set (drives the skip)

    nav_outputs = groups + rail + [back_btn, next_btn, step]

    def _set_step(target: int):
        target = max(0, min(_N - 1, int(target)))
        updates = [gr.update(visible=(k == target)) for k in range(_N)]
        updates += [gr.update(variant=("primary" if k == target else "secondary")) for k in range(_N)]
        updates += [gr.update(interactive=(target > 0)),
                    gr.update(interactive=(target < _N - 1)),
                    target]
        return updates

    # Conditional skip: when a reference image is supplied on step 1, Base Gen
    # (index 2) is skipped — the reference becomes the base.
    BASE_IDX = 2
    reference = comps["info"]["reference_image"]

    def _nav(s, ref_on, delta):
        target = int(s) + delta
        if target == BASE_IDX and ref_on:
            target = BASE_IDX + delta  # 3 going forward, 1 going back
        return _set_step(target)

    for i, btn in enumerate(rail):
        btn.click(lambda i=i: _set_step(i), outputs=nav_outputs)
    # Nav reads the has_ref boolean State (NOT the reference Image) to avoid Gradio
    # ImageData validation on every Back/Next click.
    back_btn.click(lambda s, r: _nav(s, r, -1), inputs=[step, has_ref], outputs=nav_outputs)
    next_btn.click(lambda s, r: _nav(s, r, +1), inputs=[step, has_ref], outputs=nav_outputs)

    # A reference image becomes the base (so Face/Body has something to work on),
    # greys Base Gen on the rail, and flips has_ref.
    base_sel = comps["base"]["selected_base"]

    def _ref_changed(ref):
        on = bool(ref)
        rail_upd = (gr.update(value="③ Base (skipped)", interactive=False) if on
                    else gr.update(value=STEPS[BASE_IDX][1], interactive=True))
        return rail_upd, (ref if on else gr.update()), on

    reference.change(_ref_changed, inputs=[reference],
                     outputs=[rail[BASE_IDX], base_sel, has_ref])

    # Keep the Face/Body preview mirroring the current base (ref / gen / swap result).
    base_sel.change(lambda p: p or gr.update(), inputs=[base_sel],
                    outputs=[comps["swap"]["base_preview"]])

    _wire_load_save(comps, settings, poses_state)
    _wire_persistence(comps, settings, poses_state, init or {})

    return {"step": step, "groups": groups, "rail": rail, "nav": (back_btn, next_btn),
            "components": comps, "settings": settings, "prereqs": prereqs,
            "poses_state": poses_state}


# Fields autosaved/restored. Keyed "<group>.<name>"; "settings.*" come from the bar.
# Image components are intentionally NOT persisted: their values are temp file
# paths (gone after restart) and restoring a bare string into a gr.Image that is
# also an event input breaks Gradio's ImageData validation.
_PERSIST_SPEC = {
    "info": ["name", "description", "style"],
    "prompt": ["positive_prompt", "negative_prompt"],
    "base": ["count"],
    "settings": ["model", "sampler", "scheduler", "steps", "cfg_scale", "clip_skip",
                 "seed", "width", "height", "adetailer", "loras", "lora_multipliers"],
    "swap": ["face_enhancer", "face_enhancer_strength", "face_blend_ratio",
             "body_ip_scale", "body_denoise", "body_cfg", "body_cn_strength"],
    "poses": ["ref_look_strength", "apply_body_to_poses"],
    "train": ["dataset", "low_vram", "epochs"],
}
_IMAGE_KEYS = set()  # no image fields persisted


def _wire_persistence(comps, settings, poses_state, init):
    pairs = []  # (key, component)
    for group, names in _PERSIST_SPEC.items():
        src = settings if group == "settings" else comps.get(group, {})
        for name in names:
            comp = src.get(name)
            if comp is not None:
                pairs.append((f"{group}.{name}", comp))

    keys = [k for k, _ in pairs]
    fields = [c for _, c in pairs]
    defaults = {k: c.value for k, c in pairs}  # builder defaults (captured pre-restore)

    # Restore persisted values (skip image paths whose temp file is gone).
    for k, c in pairs:
        if k in init:
            v = init[k]
            if k in _IMAGE_KEYS and not (isinstance(v, str) and os.path.isfile(v)):
                continue
            c.value = v

    def _save(*vals):
        wizard_state.save(dict(zip(keys, vals)))

    for c in fields:
        c.change(_save, inputs=fields, outputs=[])

    # Clear Wizard: reset every field to its builder default + wipe persisted state.
    clear_btn = comps["info"].get("clear_btn")
    if clear_btn is not None:
        def _clear():
            wizard_state.clear()
            return [defaults[k] for k in keys] + [{"poses": [], "specs": []}]
        clear_btn.click(_clear, outputs=fields + [poses_state])


def _summary(cs, cdir) -> str:
    lines = [f"**{cs.name}** — {cs.style}", f"`{cdir}`", ""]
    lines.append(f"- Description: {cs.description or '—'}")
    lines.append(f"- Base image: {'✅' if cs.selected_base else '—'}")
    lines.append(f"- Face source: {'✅' if cs.face_source_path else '—'}  ·  "
                 f"Body source: {'✅' if cs.body_source_path else '—'}")
    lines.append(f"- Approved poses: {len(cs.approved_poses)}")
    lines.append(f"- Trigger word: `{cs.trigger}`")
    return "\n".join(lines)


def _wire_load_save(comps, settings, poses_state):
    info, prm, base, swap = (comps["info"], comps["prompt"],
                             comps["base"], comps["swap"])
    save = comps["save"]

    info["refresh_btn"].click(lambda: gr.update(choices=paths.list_characters()),
                              outputs=[info["load_existing"]])

    load_outputs = [info["name"], info["description"], info["style"],
                    prm["positive_prompt"], prm["negative_prompt"],
                    info["reference_image"], base["selected_base"],
                    settings["steps"], settings["cfg_scale"], settings["seed"],
                    settings["width"], settings["height"], settings["adetailer"]]

    def _load(sel):
        if not sel:
            return [gr.update()] * len(load_outputs)
        cs = character.load_character(paths.character_dir(sel))
        return [cs.name, cs.description, cs.style, cs.positive_prompt, cs.negative_prompt,
                cs.reference_image or None, cs.selected_base or None,
                cs.steps, cs.cfg_scale, cs.seed, cs.width, cs.height, cs.adetailer]

    info["load_btn"].click(_load, inputs=[info["load_existing"]], outputs=load_outputs)

    # Seed the positive prompt from description + style; fill default negative if empty.
    def _seed(desc, style, cur_neg):
        neg = cur_neg if (cur_neg and cur_neg.strip()) else character.DEFAULT_NEGATIVE
        return character.build_seed_prompt(desc, style), neg

    prm["seed_prompt"].click(_seed,
        inputs=[info["description"], info["style"], prm["negative_prompt"]],
        outputs=[prm["positive_prompt"], prm["negative_prompt"]])

    save_inputs = [info["name"], info["description"], info["style"],
                   prm["positive_prompt"], prm["negative_prompt"],
                   info["reference_image"], base["selected_base"],
                   swap["face_source"], swap["body_source"],
                   settings["steps"], settings["cfg_scale"], settings["seed"],
                   settings["width"], settings["height"], settings["adetailer"],
                   poses_state]

    def _save(name, desc, style, pos, neg, ref, sbase, face_src, body_src,
              steps, cfg, seed, width, height, adet, poses_data,
              progress=gr.Progress()):
        if not (name and name.strip()):
            return "⚠️ Enter a character name first.", gr.update()
        pd = poses_data or {}
        cs = character.CharacterState(
            name=name, description=desc or "", style=style,
            positive_prompt=pos or "", negative_prompt=neg or "",
            reference_image=ref or "", selected_base=(sbase or ref or ""),
            face_source_path=face_src or "", body_source_path=body_src or "",
            face_swap_enabled=bool(face_src), body_swap_enabled=bool(body_src),
            approved_poses=list(pd.get("poses", [])),
            approved_pose_specs=list(pd.get("specs", [])),
            steps=int(steps), cfg_scale=float(cfg), seed=int(seed),
            width=int(width), height=int(height), adetailer=bool(adet))
        cdir = paths.character_dir(name)
        character.save_character(cdir, cs)
        msg = f"✅ Saved to `{cdir}`"
        if cs.approved_poses:
            try:
                progress(0.5, desc="Building LoRA datasets…")
                ddir = paths.character_dataset_dir(name)
                datasets.build_character_datasets(ddir, cs)
                msg += f" · datasets built at `{ddir}`"
            except Exception:
                msg += " · ⚠️ dataset build failed (see console)"
                traceback.print_exc()
        return msg, _summary(cs, cdir)

    save["save"].click(_save, inputs=save_inputs,
                       outputs=[save["save_status"], save["summary"]])
