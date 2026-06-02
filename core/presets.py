"""Recommended generation settings per model family.

When a model is selected the Generation-settings bar auto-populates with that
family's recommended cfg / steps / sampler / scheduler / clip-skip + portrait
resolution. Three layers, highest priority first:

  1. User overrides — edited in OrphanSuite → "Default Generation Values" and
     persisted to the cross-plugin ``.orphansuite.json`` (so every saintorphan
     plugin shares them). Per family.
  2. Native model defaults — for Flux/Z-Image/Qwen, the model's own Wan2GP
     defaults (num_inference_steps / guidance_scale / resolution) are more
     accurate per-model than a generic family number (e.g. Flux 2 Klein is 4-step,
     Z-Image Turbo 8-step), so they win over the factory family numbers.
  3. Factory family defaults — the curated starting points below.

A preset is a plain dict with any of FIELDS. Missing keys mean "leave as-is".
"""
from __future__ import annotations

from pathlib import Path

from . import discovery

FIELDS = ["steps", "cfg", "sampler", "scheduler", "clip_skip", "width", "height"]
# Editor order (native families first, then SD-family).
FAMILIES = ["Flux", "Z-Image", "Qwen", "SDXL", "Pony", "Illustrious"]
_NATIVE_FAMILIES = {"Flux", "Z-Image", "Qwen"}
_CONFIG_KEY = "gen_defaults"  # key in .orphansuite.json (shared across plugins)

# Curated factory starting points (portrait, for a character lab). Native pipelines
# own their sampler/scheduler so those are neutral ("default"/""); per-model step/
# cfg/resolution still override these at selection time (layer 2 above).
FACTORY = {
    "Flux":        {"steps": 20, "cfg": 3.5, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "Z-Image":     {"steps": 8, "cfg": 1.0, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "Qwen":        {"steps": 20, "cfg": 4.0, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "SDXL":        {"steps": 30, "cfg": 7.0, "sampler": "DPM++ 2M",
                    "scheduler": "Karras", "clip_skip": 2, "width": 832, "height": 1216},
    "Pony":        {"steps": 28, "cfg": 7.0, "sampler": "DPM++ 2M SDE",
                    "scheduler": "Karras", "clip_skip": 2, "width": 832, "height": 1216},
    "Illustrious": {"steps": 28, "cfg": 6.0, "sampler": "Euler a",
                    "scheduler": "Normal", "clip_skip": 2, "width": 896, "height": 1152},
}

# Keep within the UI slider bounds (settings_bar.py).
_W_MIN, _W_MAX, _STEPS_MAX, _CFG_MIN, _CFG_MAX = 256, 2048, 60, 1.0, 15.0


def _snap(v, lo=_W_MIN, hi=_W_MAX, step=64) -> int:
    return max(lo, min(hi, int(round(v / step) * step)))


def is_native_family(family: str) -> bool:
    return family in _NATIVE_FAMILIES


def family_of(model_value: str) -> str | None:
    """Family name for a dropdown value: SD → SDXL/Pony/Illustrious; native →
    Flux/Z-Image/Qwen; None if no model."""
    backend, ident = discovery.parse_model_value(model_value)
    if backend == "sd":
        return discovery.categorize_sdxl(Path(ident).stem)
    if backend == "native":
        return discovery.categorize_native(ident)
    return None


# --- persisted user overrides ----------------------------------------------

def user_overrides() -> dict:
    """All per-family overrides from the shared config: {family: {field: value}}."""
    from . import paths
    d = paths.get_shared(_CONFIG_KEY, {})
    return d if isinstance(d, dict) else {}


def set_overrides(family: str, values: dict) -> None:
    """Persist a family's override (only known FIELDS, non-None) to shared config."""
    from . import paths
    cur = dict(user_overrides())
    cur[family] = {k: values[k] for k in FIELDS if values.get(k) is not None}
    paths.set_shared(_CONFIG_KEY, cur)


def clear_overrides(family: str) -> None:
    """Drop a family's override → reverts to factory."""
    from . import paths
    cur = dict(user_overrides())
    if family in cur:
        cur.pop(family, None)
        paths.set_shared(_CONFIG_KEY, cur)


def has_override(family: str) -> bool:
    return bool(user_overrides().get(family))


# --- resolved values -------------------------------------------------------

def factory(family: str) -> dict:
    """Concrete factory defaults for a family (what the editor shows by default)."""
    return dict(FACTORY.get(family, FACTORY["SDXL"]))


def _native_from_model(model_value, get_default_settings) -> dict:
    """A native model's own Wan2GP defaults → preset dict (portrait-oriented)."""
    backend, ident = discovery.parse_model_value(model_value)
    out: dict = {}
    if backend != "native" or not callable(get_default_settings):
        return out
    try:
        d = dict(get_default_settings(ident) or {})
    except Exception:
        return out
    steps = d.get("num_inference_steps")
    if isinstance(steps, (int, float)) and steps > 0:
        out["steps"] = int(min(steps, _STEPS_MAX))
    g = d.get("guidance_scale")
    if isinstance(g, (int, float)):
        out["cfg"] = float(max(_CFG_MIN, min(_CFG_MAX, g))) if g else _CFG_MIN
    try:
        w, h = str(d.get("resolution")).lower().split("x")
        lo, hi = sorted((_snap(int(w)), _snap(int(h))))  # portrait
        out["width"], out["height"] = lo, hi
    except Exception:
        pass
    return out


def effective(family: str) -> dict:
    """Factory family defaults with the user's saved override layered on (no
    per-model data) — used by the editor to show the current default."""
    base = factory(family)
    base.update(user_overrides().get(family) or {})
    return base


def for_model(model_value, get_default_settings=None) -> dict:
    """Recommended settings to apply on model selection. {} if no model.
    Priority: user override > native per-model defaults > factory family."""
    fam = family_of(model_value)
    if not fam:
        return {}
    base = factory(fam)
    if is_native_family(fam):
        base.update(_native_from_model(model_value, get_default_settings))
    base.update(user_overrides().get(fam) or {})  # user choice wins
    return base


# --- resolution (locked to the family's trained portrait aspect) -----------

def _portrait_base(model_value, get_default_settings=None):
    """The recommended PORTRAIT (w, h) for a model's family — width <= height.
    Uses the family PORTRAIT preset (factory + user override), NOT the model's own
    native default: some native models (e.g. Z-Image) default to 1:1, which would
    otherwise collapse Replicant's portrait base to square. Steps/CFG still come
    from the model via for_model; only the aspect/size is family-portrait here."""
    fam = family_of(model_value)
    rec = effective(fam) if fam else {}
    w, h = rec.get("width") or 832, rec.get("height") or 1216
    return min(w, h), max(w, h)


def recommended_resolution(model_value, get_default_settings=None) -> str:
    """The recommended portrait resolution as a 'WxH' string."""
    pw, ph = _portrait_base(model_value, get_default_settings)
    return f"{pw}x{ph}"


def resolution_tiers(model_value, get_default_settings=None) -> list:
    """Dropdown choices [(label, 'WxH')] of portrait resolutions at the family's
    trained aspect — a few size tiers around the recommended one. The user can pick
    a smaller/larger size but NOT a different aspect (Replicant locks aspect)."""
    pw, ph = _portrait_base(model_value, get_default_settings)
    seen, tiers = set(), []
    for scale in (0.66, 0.83, 1.0, 1.2, 1.4):
        w, h = _snap(pw * scale), _snap(ph * scale)
        if (w, h) not in seen:
            seen.add((w, h))
            tiers.append((w, h))
    if (pw, ph) not in seen:
        tiers.append((pw, ph))
    tiers.sort(key=lambda wh: wh[0] * wh[1])
    return [(f"{w}×{h}" + (" — recommended" if (w, h) == (pw, ph) else ""), f"{w}x{h}")
            for (w, h) in tiers]


def oriented(resolution_str, orientation="portrait") -> tuple:
    """A 'WxH' portrait resolution → (w, h) for a pose orientation:
    portrait (as-is), landscape (swapped), or square (1:1 at ~the same area)."""
    import math
    try:
        a, b = str(resolution_str).lower().split("x")
        a, b = int(a), int(b)
    except Exception:
        a, b = 832, 1216
    pw, ph = min(a, b), max(a, b)
    if orientation == "landscape":
        return ph, pw
    if orientation == "square":
        s = _snap(math.sqrt(pw * ph))
        return s, s
    return pw, ph
