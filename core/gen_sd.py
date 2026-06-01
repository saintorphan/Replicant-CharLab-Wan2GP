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

from . import models, paths

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


def _free_torch():
    import gc
    import torch
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def release_sd():
    """Unload the cached SD txt2img pipeline + free its VRAM (the SDXL checkpoint
    is ~6.5GB). Call before a different heavy model needs the GPU."""
    global _pipeline
    try:
        if _pipeline is not None:
            _pipeline.unload()
    except Exception:
        logger.debug("SD unload failed", exc_info=True)
    _free_torch()


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
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    with models.no_auto_download():  # never silently pull weights/configs
        pipe.load(checkpoint_path)
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


def generate_img2img(checkpoint_path, image_path, prompt, negative, width, height,
                     steps, cfg, seed, denoise=0.6, sampler="DPM++ 2M", scheduler="Karras",
                     batch_size=1, clip_skip=1, out_dir=None) -> list[str]:
    """Reimagine an init image with an SD-family checkpoint (img2img)."""
    import random as _random
    pipe = get_pipeline()
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    with models.no_auto_download():
        pipe.load(checkpoint_path)
        images = pipe.generate_img2img(
            image=image_path, prompt=prompt, negative_prompt=negative or "",
            denoising_strength=float(denoise), width=int(width), height=int(height),
            steps=int(steps), cfg_scale=float(cfg), seed=int(seed), sampler=sampler,
            scheduler=scheduler, batch_size=int(batch_size), clip_skip=int(clip_skip))
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"i2i_{int(seed)}_{i}.png"
        try:
            img.save(f); saved.append(str(f))
        except Exception:
            logger.warning("failed saving img2img %d", i, exc_info=True)
    return saved


class _ProjMgr:
    """Minimal project manager for ImagePipeline — only get_project_path is used."""
    def __init__(self, root):
        self._root = Path(root)

    def get_project_path(self, name):
        p = self._root / (name or "replicant")
        p.mkdir(parents=True, exist_ok=True)
        return p


def body_swap(checkpoint_path, base_path, source_person_path, prompt, negative,
              cn_strength=0.7, ip_scale=0.8, denoise=0.75, cfg=7.0, steps=30,
              seed=-1, sampler="DPM++ 3M SDE", scheduler="Karras",
              adetailer=True, progress=None) -> str | None:
    """Body double: segment the person in the base, extract an openpose control
    image, then inpaint a new body with the source person's identity (ControlNet
    openpose + IP-Adapter faceid_plus). SD-family checkpoints only.

    Ported from SupremeDiffusion's BodyDoubleWorker orchestration. Heavy: pulls
    BiRefNet (segment), openpose annotator, ControlNet-openpose and IP-Adapter
    (auto-downloaded). Returns the result image path."""
    import os
    from PIL import Image
    from . import deps
    deps.ensure_body_swap(progress)  # auto-install controlnet_aux/kornia/ultralytics if missing
    _import_pipeline_cls()  # ensure SD_CHECKOUT on sys.path

    def _say(frac, msg):
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except Exception:
                pass
    from supremediffusion.core.image_pipeline import ImageGenerationPipeline
    from supremediffusion.config.project_config import ProjectConfig
    from supremediffusion.models.segmentation import (segment_foreground,
                                                       release_segmentation_model)
    from supremediffusion.models.controlnet_types import (run_preprocessor,
                                                          preprocessor_overrides_for)

    models_root = str(Path(SD_CHECKOUT) / "models")
    cfg_obj = ProjectConfig()
    cfg_obj.bodydouble_checkpoint = checkpoint_path
    cfg_obj.bodydouble_controlnet_type = "openpose"
    cfg_obj.bodydouble_controlnet_strength = float(cn_strength)
    cfg_obj.bodydouble_ip_adapter_variant = "faceid_plus"
    cfg_obj.bodydouble_ip_adapter_scale = float(ip_scale)
    cfg_obj.bodydouble_denoising_strength = float(denoise)
    cfg_obj.bodydouble_cfg_scale = float(cfg)
    cfg_obj.bodydouble_steps = int(steps)
    cfg_obj.bodydouble_sampler = sampler or "DPM++ 3M SDE"
    cfg_obj.bodydouble_scheduler = scheduler or "Karras"
    cfg_obj.bodydouble_prompt = prompt or ""
    cfg_obj.bodydouble_negative_prompt = negative or ""
    cfg_obj.bodydouble_seed = int(seed)
    # ADetailer face restore on the result (best-effort; honored if the pipeline reads it).
    cfg_obj.bodydouble_adetailer = bool(adetailer)
    cfg_obj.adetailer = bool(adetailer)

    with models.no_auto_download():  # never silently pull BiRefNet/ControlNet/IP-Adapter
        release_sd()  # free the txt2img checkpoint so BiRefNet/body-double fit
        _say(0.15, "Segmenting the person (BiRefNet)…")
        mask = segment_foreground(base_path, models_root)
        release_segmentation_model()  # free BiRefNet before the SD body-double loads
        _free_torch()
        try:
            _say(0.4, "Extracting pose (OpenPose)…")
            target_img = Image.open(base_path).convert("RGB")
            control = [run_preprocessor("openpose", target_img,
                                        **preprocessor_overrides_for("openpose", cfg_obj))]
            _say(0.6, "Generating body double (ControlNet + IP-Adapter)…")
            facade = ImageGenerationPipeline(get_pipeline(), _ProjMgr(paths.cache_dir() / "body_swap"))
            results = facade.run_body_double(
                project_name="replicant", target_image=base_path, mask=mask,
                source_person=source_person_path, control_images=control,
                config=cfg_obj, num_images=1)
            return results[0] if results else None
        finally:
            try:
                os.unlink(mask)
            except Exception:
                pass
            try:  # free body-double controlnet/ip-adapter VRAM for the next step
                p = get_pipeline()
                if hasattr(p, "unload_body_double"):
                    p.unload_body_double()
                if hasattr(p, "unload_controlnet"):
                    p.unload_controlnet()
            except Exception:
                pass
            _free_torch()
