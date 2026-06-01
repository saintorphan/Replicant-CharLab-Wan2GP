"""The wizard shell: logo banner, clickable step rail, Back/Next nav over seven
visibility-toggled step panels (Gradio 5.29 has no native stepper component)."""
from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

from ..core import character, paths
from .prereqs import build_prereqs
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


def build_wizard():
    """Build the wizard UI inside the current tab context.

    Returns a dict with ``step`` (gr.State), ``groups`` (list), ``rail`` (list of
    buttons), ``nav`` (back/next buttons), and ``components`` (per-step widget
    dicts keyed by step id) so the plugin can wire generation/save logic."""
    gr.HTML(_banner_html())

    # Prerequisites (directories + models) ----------------------------------
    prereqs = build_prereqs()

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

    def _nav(s, ref, delta):
        target = int(s) + delta
        if target == BASE_IDX and ref:
            target = BASE_IDX + delta  # 3 going forward, 1 going back
        return _set_step(target)

    for i, btn in enumerate(rail):
        btn.click(lambda i=i: _set_step(i), outputs=nav_outputs)
    back_btn.click(lambda s, ref: _nav(s, ref, -1), inputs=[step, reference], outputs=nav_outputs)
    next_btn.click(lambda s, ref: _nav(s, ref, +1), inputs=[step, reference], outputs=nav_outputs)

    # Reflect the skip on the rail: grey out Base Gen when a reference is set.
    def _ref_changed(ref):
        if ref:
            return gr.update(value="③ Base (skipped)", interactive=False)
        return gr.update(value=STEPS[BASE_IDX][1], interactive=True)

    reference.change(_ref_changed, inputs=[reference], outputs=[rail[BASE_IDX]])

    _wire_load_save(comps)

    return {"step": step, "groups": groups, "rail": rail,
            "nav": (back_btn, next_btn), "components": comps, "prereqs": prereqs}


def _summary(cs, cdir) -> str:
    lines = [f"**{cs.name}** — {cs.style}", f"`{cdir}`", ""]
    lines.append(f"- Description: {cs.description or '—'}")
    lines.append(f"- Base image: {'✅' if cs.selected_base else '—'}")
    lines.append(f"- Face source: {'✅' if cs.face_source_path else '—'}  ·  "
                 f"Body source: {'✅' if cs.body_source_path else '—'}")
    lines.append(f"- Approved poses: {len(cs.approved_poses)}")
    lines.append(f"- Trigger word: `{cs.trigger}`")
    return "\n".join(lines)


def _wire_load_save(comps):
    info, prm, base, swap = (comps["info"], comps["prompt"],
                             comps["base"], comps["swap"])
    save = comps["save"]

    info["refresh_btn"].click(lambda: gr.update(choices=paths.list_characters()),
                              outputs=[info["load_existing"]])

    load_outputs = [info["name"], info["description"], info["style"],
                    prm["positive_prompt"], prm["negative_prompt"],
                    info["reference_image"], base["selected_base"],
                    base["steps"], base["cfg_scale"], base["seed"],
                    base["width"], base["height"], base["adetailer"]]

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
                   base["steps"], base["cfg_scale"], base["seed"],
                   base["width"], base["height"], base["adetailer"]]

    def _save(name, desc, style, pos, neg, ref, sbase, face_src, body_src,
              steps, cfg, seed, width, height, adet):
        if not (name and name.strip()):
            return "⚠️ Enter a character name first.", gr.update()
        cs = character.CharacterState(
            name=name, description=desc or "", style=style,
            positive_prompt=pos or "", negative_prompt=neg or "",
            reference_image=ref or "", selected_base=(sbase or ref or ""),
            face_source_path=face_src or "", body_source_path=body_src or "",
            face_swap_enabled=bool(face_src), body_swap_enabled=bool(body_src),
            steps=int(steps), cfg_scale=float(cfg), seed=int(seed),
            width=int(width), height=int(height), adetailer=bool(adet))
        cdir = paths.character_dir(name)
        character.save_character(cdir, cs)
        return f"✅ Saved to `{cdir}`", _summary(cs, cdir)

    save["save"].click(_save, inputs=save_inputs,
                       outputs=[save["save_status"], save["summary"]])
