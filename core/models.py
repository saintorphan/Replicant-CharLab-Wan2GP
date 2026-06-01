"""Registry of models this extension needs that Wan2GP does NOT bundle.

The wizard's Models panel reads this registry to show per-model status and a
Download button. Downloads stash into ``paths.models_dir()`` at each entry's
``subpath``.

URLs marked ``url=None`` still need a confirmed source — the Download button is
disabled for them until one is filled in (better than shipping a wrong link).
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import paths

logger = logging.getLogger("replicant.models")


@dataclass
class ModelSpec:
    key: str
    name: str
    subpath: str          # relative to models_dir()
    purpose: str
    required: bool
    url: str | None = None
    note: str = ""

    def local_path(self) -> Path:
        return paths.models_dir() / self.subpath

    def is_present(self) -> bool:
        return self.local_path().is_file()


# buffalo_l (InsightFace face detection) is intentionally NOT listed: insightface
# auto-downloads it to ~/.insightface on first use.
REGISTRY: list[ModelSpec] = [
    ModelSpec(
        key="inswapper_128", name="InSwapper 128 (face swap)",
        subpath="face/inswapper_128.onnx",
        purpose="Face swap onto the base image + base-face→poses identity lock.",
        required=True,
        url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
    ),
    ModelSpec(
        key="gfpgan", name="GFPGAN v1.4 (face enhancer)",
        subpath="face/GFPGANv1.4.onnx",
        purpose="Optional face restoration after swaps.",
        required=False,
        url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx",
    ),
    ModelSpec(
        key="codeformer", name="CodeFormer (face enhancer)",
        subpath="face/codeformer.onnx",
        purpose="Optional face restoration after swaps.",
        required=False,
        url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
    ),
    ModelSpec(
        key="face_yolov8s", name="ADetailer face_yolov8s",
        subpath="face/face_yolov8s.pt",
        purpose="Better face detection for restoration on tough angles.",
        required=False,
        url="https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
    ),
]


def by_key(key: str) -> ModelSpec | None:
    return next((m for m in REGISTRY if m.key == key), None)


def status() -> list[dict]:
    return [{"key": m.key, "name": m.name, "present": m.is_present(),
             "required": m.required, "downloadable": bool(m.url),
             "path": str(m.local_path()), "purpose": m.purpose, "note": m.note}
            for m in REGISTRY]


def download(key: str, progress=None) -> str:
    """Download a registry model into models_dir(). Returns a status string."""
    spec = by_key(key)
    if spec is None:
        return f"[Error] Unknown model '{key}'."
    if not spec.url:
        return f"[Error] No confirmed download URL for {spec.name} yet."
    dst = spec.local_path()
    if dst.is_file():
        return f"[OK] {spec.name} already present."
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        def _hook(blocks, bsize, total):
            if progress and total > 0:
                progress(min(1.0, blocks * bsize / total), desc=f"Downloading {spec.name}")
        urllib.request.urlretrieve(spec.url, tmp, _hook)
        tmp.replace(dst)
        return f"[Success] {spec.name} → {dst}"
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        logger.warning("download failed for %s", key, exc_info=True)
        return f"[Error] Download failed for {spec.name}: {e}"
