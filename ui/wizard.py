"""The wizard shell: logo banner, clickable step rail, Back/Next nav over seven
visibility-toggled step panels (Gradio 5.29 has no native stepper component)."""
from __future__ import annotations

import base64
import os
import traceback
from pathlib import Path

import gradio as gr

from ..core import character, datasets, gen_sd, paths, wizard_state
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
    # Header: taglines (left) · banner (center) · Clear Wizard (right) -------
    with gr.Row(elem_id="replicant-header"):
        with gr.Column(scale=3):
            gr.HTML('<div class="replicant-taglines">'
                    '<div class="replicant-tagline">&ldquo;Generate, modify, and train '
                    'image and video<br>character LoRAs from a single prompt or '
                    'reference!&rdquo;</div>'
                    '<div class="replicant-tagline">&ldquo;Wan 2.2, LTX 2.3, Z-Image, '
                    'Flux, and...<br>wait, is that SDXL? Illustrious?? PONY?!&rdquo;</div>'
                    '</div>')
        with gr.Column(scale=10):
            gr.HTML(_banner_html())
        with gr.Column(scale=3, min_width=200, elem_id="replicant-clearcol"):
            header_clear_btn = gr.Button("🗑 Clear Wizard", variant="stop",
                                         scale=0, min_width=150)
            header_clear_files = gr.Checkbox(value=False,
                label="Delete all unsaved generations (candidates / base / reference / "
                      "swaps / poses)")
            gr.HTML('<div class="replicant-ghlink"><a href="https://github.com/'
                    'saintorphan/Replicant-CharLab-Wan2GP" target="_blank" '
                    'rel="noopener">⧉ GitHub repo</a></div>')

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
        g, c = builder(visible=(i == 0), init=init)
        groups.append(g)
        comps[STEPS[i][0]] = c
    # Clear Wizard now lives in the header; expose it where the wiring looks.
    comps["setup"]["clear_btn"] = header_clear_btn
    comps["setup"]["clear_files"] = header_clear_files

    # Nav -------------------------------------------------------------------
    with gr.Row(elem_id="replicant-nav"):
        back_btn = gr.Button("◀ Back", interactive=False)
        next_btn = gr.Button("Next ▶", variant="primary")

    step = gr.State(0)
    poses_state = gr.State({"poses": [], "specs": []})  # filled by pose gen (step 5)

    sw, sh = settings["width"], settings["height"]
    nav_outputs = groups + rail + [back_btn, next_btn, step, sw, sh]
    _BASE_STEP = 1  # ② Base Gen — forced portrait + locked dims

    def _set_step(target: int):
        target = max(0, min(_N - 1, int(target)))
        updates = [gr.update(visible=(k == target)) for k in range(_N)]
        updates += [gr.update(variant=("primary" if k == target else "secondary")) for k in range(_N)]
        updates += [gr.update(interactive=(target > 0)),
                    gr.update(interactive=(target < _N - 1)),
                    target]
        if target == _BASE_STEP:  # base must be a full-body portrait
            updates += [gr.update(value=832, interactive=False),
                        gr.update(value=1216, interactive=False)]
        else:
            updates += [gr.update(interactive=True), gr.update(interactive=True)]
        return updates

    reference = comps["setup"]["reference_image"]

    # Base Gen is no longer skipped for a reference — step 3 shows the reference and
    # offers Reimagine (img2img) or skip-as-base.
    def _nav(s, delta):
        return _set_step(int(s) + delta)

    for i, btn in enumerate(rail):
        btn.click(lambda i=i: _set_step(i), outputs=nav_outputs)
    back_btn.click(lambda s: _nav(s, -1), inputs=[step], outputs=nav_outputs)
    next_btn.click(lambda s: _nav(s, +1), inputs=[step], outputs=nav_outputs)

    # A reference seeds the initial base + shows in step 3's avatar, and enables the
    # "Revert to Reference" button. (Generation never changes the base.)
    base_sel = comps["base"]["selected_base"]
    ref_avatar = comps["base"].get("ref_avatar")
    revert_ref = comps["base"].get("revert_ref")

    def _ref_changed(ref):
        return (ref if ref else gr.update()), (ref or None), gr.update(interactive=bool(ref))

    reference.change(_ref_changed, inputs=[reference],
                     outputs=[base_sel, ref_avatar, revert_ref])
    if revert_ref is not None:
        revert_ref.click(lambda ref: ref or gr.update(), inputs=[reference], outputs=[base_sel])

    # Keep the Face/Body preview mirroring the current base (ref / gen / swap result).
    base_sel.change(lambda p: p or gr.update(), inputs=[base_sel],
                    outputs=[comps["swap"]["base_preview"]])

    # Carry prompts onto Base Gen and keep them in sync both ways (settles because
    # Gradio doesn't re-fire .change when the value is unchanged).
    prm, bp, bn = comps["setup"], comps["base"]["pos"], comps["base"]["neg"]
    prm["positive_prompt"].change(lambda v: v, inputs=[prm["positive_prompt"]], outputs=[bp])
    prm["negative_prompt"].change(lambda v: v, inputs=[prm["negative_prompt"]], outputs=[bn])
    bp.change(lambda v: v, inputs=[bp], outputs=[prm["positive_prompt"]])
    bn.change(lambda v: v, inputs=[bn], outputs=[prm["negative_prompt"]])

    _wire_load_save(comps, settings, poses_state)
    _wire_persistence(comps, settings, poses_state, init or {})

    return {"step": step, "groups": groups, "rail": rail, "nav": (back_btn, next_btn),
            "components": comps, "settings": settings, "prereqs": prereqs,
            "poses_state": poses_state}


# SCALAR fields autosaved/restored via post-hoc .value. Keyed "<group>.<name>".
_PERSIST_SPEC = {
    "setup": ["name", "description", "style", "positive_prompt", "negative_prompt"],
    "base": ["count", "denoise"],
    "settings": ["model", "sampler", "scheduler", "steps", "cfg_scale", "clip_skip",
                 "seed", "width", "height", "adetailer", "loras", "lora_multipliers"],
    "swap": ["face_enhancer", "face_enhancer_strength", "face_blend_ratio",
             "face_adet_pos", "face_adet_neg",
             "body_ip_scale", "body_denoise", "body_cfg", "body_cn_strength",
             "adetailer", "body_adet_pos", "body_adet_neg"],
    "poses": ["ref_look_strength", "apply_body_to_poses"],
    "train": ["dataset", "low_vram", "epochs"],
}
# IMAGE/gallery fields persisted separately: copied to a stable dir and restored
# via the component CONSTRUCTOR (post-hoc .value on a gr.Image breaks ImageData
# validation when it's also an event input). Key -> (group, name).
_IMAGE_FIELDS = [("setup.reference_image", "setup", "reference_image"),
                 ("base.selected_base", "base", "selected_base"),
                 ("swap.face_source", "swap", "face_source"),
                 ("swap.body_source", "swap", "body_source")]
_GALLERY_FIELDS = [("base.candidates", "base", "candidates")]


def _persist_dir():
    return paths.cache_dir() / "persist"


def _extract_path(item):
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and item:
        return _extract_path(item[0])
    if isinstance(item, dict):
        return item.get("path") or (item.get("image") or {}).get("path") or item.get("name")
    return None


def _stable_image(path, key):
    import shutil
    if not (isinstance(path, str) and os.path.isfile(path)):
        return None
    d = _persist_dir(); d.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    dst = d / (key.replace(".", "_") + ext)
    try:
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy2(path, dst)
        return str(dst)
    except Exception:
        return path


def _stable_gallery(vals):
    import shutil
    d = _persist_dir() / "candidates"
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    out = []
    for i, item in enumerate(vals or []):
        p = _extract_path(item)
        if p and os.path.isfile(p):
            dst = d / f"cand_{i:02d}{os.path.splitext(p)[1] or '.png'}"
            try:
                shutil.copy2(p, dst); out.append(str(dst))
            except Exception:
                out.append(p)
    return out


def _wire_persistence(comps, settings, poses_state, init):
    # --- scalar fields ---
    pairs = []
    for group, names in _PERSIST_SPEC.items():
        src = settings if group == "settings" else comps.get(group, {})
        for name in names:
            comp = src.get(name)
            if comp is not None:
                pairs.append((f"{group}.{name}", comp))
    keys = [k for k, _ in pairs]
    fields = [c for _, c in pairs]
    defaults = {k: c.value for k, c in pairs}
    for k, c in pairs:
        if k in init:
            c.value = init[k]

    def _save_scalars(*vals):
        d = wizard_state.load()
        d.update(dict(zip(keys, vals)))
        wizard_state.save(d)
    for c in fields:
        c.change(_save_scalars, inputs=fields, outputs=[])

    # --- image fields (copied to a stable dir; restored via constructor) ---
    img_comps = []
    for key, group, name in _IMAGE_FIELDS:
        comp = comps.get(group, {}).get(name)
        if comp is None:
            continue
        img_comps.append(comp)

        def _save_img(path, _k=key):
            d = wizard_state.load(); d[_k] = _stable_image(path, _k); wizard_state.save(d)
        comp.change(_save_img, inputs=[comp], outputs=[])

    # --- candidate gallery ---
    gal = comps.get("base", {}).get("candidates")
    if gal is not None:
        def _save_gal(vals):
            d = wizard_state.load(); d["base.candidates"] = _stable_gallery(vals)
            wizard_state.save(d)
        gal.change(_save_gal, inputs=[gal], outputs=[])

    # --- Clear Wizard: reset everything + wipe persisted state/files ---
    clear_btn = comps["setup"].get("clear_btn")
    clear_files = comps["setup"].get("clear_files")
    if clear_btn is not None:
        import shutil
        clear_outputs = fields + img_comps + ([gal] if gal is not None else []) + [poses_state]

        def _clear(delete_files):
            wizard_state.clear()  # always reset the form state
            if delete_files:      # also delete the unsaved generation files on disk
                for sub in ("persist", "sd_gen", "poses", "swap"):
                    shutil.rmtree(paths.cache_dir() / sub, ignore_errors=True)
            return ([defaults[k] for k in keys]
                    + [None] * len(img_comps) + ([None] if gal is not None else [])
                    + [{"poses": [], "specs": []}])
        clear_btn.click(_clear, inputs=[clear_files], outputs=clear_outputs)


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
    info, prm, base, swap = (comps["setup"], comps["setup"],
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
                gen_sd.release_sd()  # free any resident SD model before crop detection
                ddir = paths.character_dataset_dir(name)
                datasets.build_character_datasets(ddir, cs)
                gen_sd._free_torch()  # release the crop detector's VRAM afterward
                msg += f" · datasets built at `{ddir}`"
            except Exception:
                msg += " · ⚠️ dataset build failed (see console)"
                traceback.print_exc()
        return msg, _summary(cs, cdir)

    save["save"].click(_save, inputs=save_inputs,
                       outputs=[save["save_status"], save["summary"]])
