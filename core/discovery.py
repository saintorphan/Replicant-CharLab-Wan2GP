"""Model + LoRA discovery for the categorized model dropdown.

Two backends:
  - native: Wan2GP image models (Flux / Z-Image / Qwen) — the list is supplied by
    the plugin at runtime (from wgp globals), categorized by model_type name here.
  - sd:     SDXL/Pony/Illustrious checkpoints scanned from paths.sdxl_models_dir().
            Architecture is SDXL for all; Pony/Illustrious are name-based.

Dropdown choices are (label, value) pairs; value encodes the backend so the
generator can route: "native::<model_type>" or "sd::<checkpoint_path>".
"""
from __future__ import annotations

from pathlib import Path

from . import paths

_CKPT_EXTS = (".safetensors", ".ckpt")


def categorize_native(model_type: str) -> str | None:
    """Bucket a Wan2GP image model_type into a dropdown category, or None if it
    isn't an image model we surface."""
    mt = (model_type or "").lower()
    if "z_image" in mt or "zimage" in mt or "z-image" in mt:
        return "Z-Image"
    if "qwen_image" in mt or "qwen-image" in mt or mt.startswith("qwen_image"):
        return "Qwen"
    if "flux" in mt:
        return "Flux"
    return None


def categorize_sdxl(name: str) -> str:
    """Name-based bucket for an SDXL-architecture checkpoint."""
    n = (name or "").lower()
    if "pony" in n:
        return "Pony"
    if "illustrious" in n or "illust" in n or "noob" in n:
        return "Illustrious"
    return "SDXL"


def discover_sdxl_models(models_dir=None) -> list[dict]:
    """Scan the SDXL models dir for checkpoints, categorized."""
    d = Path(models_dir) if models_dir else paths.sdxl_models_dir()
    out = []
    if d and Path(d).is_dir():
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in _CKPT_EXTS:
                out.append({"backend": "sd", "category": categorize_sdxl(p.stem),
                            "name": p.stem, "path": str(p)})
    return out


def discover_sdxl_loras(loras_dir=None) -> list[dict]:
    d = Path(loras_dir) if loras_dir else paths.sdxl_loras_dir()
    out = []
    if d and Path(d).is_dir():
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in _CKPT_EXTS:
                out.append({"name": p.stem, "path": str(p)})
    return out


# Category display order in the dropdown.
_ORDER = ["Flux", "Z-Image", "Qwen", "SDXL", "Pony", "Illustrious"]


def build_model_choices(native_model_types=None, models_dir=None) -> list:
    """Return Gradio Dropdown choices [(label, value), ...] grouped by category.

    native_model_types: iterable of Wan2GP image model_type strings (from the app).
    """
    entries: list[dict] = []
    for mt in (native_model_types or []):
        cat = categorize_native(mt)
        if cat:
            entries.append({"backend": "native", "category": cat, "name": mt,
                            "value": f"native::{mt}"})
    for m in discover_sdxl_models(models_dir):
        entries.append({**m, "value": f"sd::{m['path']}"})

    entries.sort(key=lambda e: (_ORDER.index(e["category"]) if e["category"] in _ORDER
                                else len(_ORDER), e["name"].lower()))
    return [(f"{e['category']} · {e['name']}", e["value"]) for e in entries]


def parse_model_value(value: str) -> tuple[str, str]:
    """('native', model_type) or ('sd', checkpoint_path) from a dropdown value."""
    if not value:
        return "", ""
    backend, _, ident = value.partition("::")
    return backend, ident


def categorize_lora(name: str) -> str:
    """Same name-based family bucket as checkpoints (Pony/Illustrious/SDXL)."""
    return categorize_sdxl(name)


def model_family(model_value: str) -> str | None:
    """The SD family of the selected model (Pony/Illustrious/SDXL), or None for
    native models (Flux/Z-Image/Qwen) — which the SDXL LoRAs don't apply to."""
    backend, ident = parse_model_value(model_value)
    if backend == "sd":
        return categorize_sdxl(Path(ident).stem)
    return None


def lora_choices(loras_dir=None, family: str | None = None) -> list:
    """Categorized LoRA dropdown choices, optionally filtered to one family.
    SDXL/Pony/Illustrious LoRAs are NOT cross-compatible, so when a model is
    selected we show only its family."""
    out = []
    for m in discover_sdxl_loras(loras_dir):
        cat = categorize_lora(m["name"])
        if family and cat != family:
            continue
        out.append((f"{cat} · {m['name']}", m["path"]))
    return out
