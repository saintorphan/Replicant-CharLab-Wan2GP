"""SD/SDXL/Pony/Illustrious generation backend.

Wraps SupremeDiffusion's SDImagePipeline (imported from the SD checkout — Wan2GP
has no SD-family support). SDImagePipeline only needs ``config.model_paths`` with
sd_checkpoint_dir / sd_lora_dir / sd_vae_dir / sd_refiner_dir, so the shim is tiny.

Generation is GPU-heavy; callers must hold the Wan2GP GPU lock (and ideally unload
the main model) around these calls — see plugin.acquire_gpu/release_gpu.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from . import paths

logger = logging.getLogger("replicant.gen_sd")

# Where the SupremeDiffusion package lives (override with REPLICANT_SD_PATH).
SD_CHECKOUT = os.environ.get(
    "REPLICANT_SD_PATH", str(Path.home() / "Projects" / "SupremeDiffusionQt"))


class _SDConfig:
    """Minimal stand-in for SupremeDiffusion's global_config — only model_paths
    is read by SDImagePipeline."""
    def __init__(self):
        self.model_paths = {
            "sd_checkpoint_dir": str(paths.sdxl_models_dir()),
            "sd_lora_dir": str(paths.sdxl_loras_dir()),
            "sd_vae_dir": "",
            "sd_refiner_dir": "",
        }


_pipeline = None


def _import_pipeline_cls():
    if SD_CHECKOUT and SD_CHECKOUT not in sys.path:
        sys.path.insert(0, SD_CHECKOUT)
    from supremediffusion.models.sd_pipeline import SDImagePipeline
    return SDImagePipeline


def get_pipeline():
    """Lazily build the (cached) SDImagePipeline. Raises ImportError if the SD
    checkout isn't available."""
    global _pipeline
    if _pipeline is None:
        cls = _import_pipeline_cls()
        _pipeline = cls(_SDConfig())
    return _pipeline


def available() -> bool:
    """True if the SD pipeline can be imported (checkout present)."""
    try:
        _import_pipeline_cls()
        return True
    except Exception:
        return False


def generate_txt2img(checkpoint_path, prompt, negative, width, height, steps, cfg,
                     seed, sampler="DPM++ 2M", scheduler="", batch_size=1,
                     clip_skip=1, out_dir=None, callback=None) -> list[str]:
    """Generate image(s) with an SD-family checkpoint; returns saved file paths."""
    import random as _random
    pipe = get_pipeline()
    pipe.load(checkpoint_path)
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    images = pipe.generate_txt2img(
        prompt=prompt, negative_prompt=negative or "",
        width=int(width), height=int(height), steps=int(steps),
        cfg_scale=float(cfg), seed=int(seed), sampler=sampler, scheduler=scheduler,
        batch_size=int(batch_size), clip_skip=int(clip_skip), callback=callback,
    )
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"sd_{int(seed)}_{i}.png"
        try:
            img.save(f)
            saved.append(str(f))
        except Exception:
            logger.warning("failed saving SD image %d", i, exc_info=True)
    return saved
