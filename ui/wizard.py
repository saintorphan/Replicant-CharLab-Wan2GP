"""The wizard shell: logo banner, clickable step rail, Back/Next nav over seven
visibility-toggled step panels (Gradio 5.29 has no native stepper component)."""
from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

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

    for i, btn in enumerate(rail):
        btn.click(lambda i=i: _set_step(i), outputs=nav_outputs)
    back_btn.click(lambda s: _set_step(s - 1), inputs=[step], outputs=nav_outputs)
    next_btn.click(lambda s: _set_step(s + 1), inputs=[step], outputs=nav_outputs)

    return {"step": step, "groups": groups, "rail": rail,
            "nav": (back_btn, next_btn), "components": comps}
