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


# ---- Shared IP-Adapter masked-inpaint identity transfer -------------------
# One primitive for both: body swap (ref = source body) and pose identity
# (ref = base). diffusers SDXL inpaint + IP-Adapter (plus, ViT-H). No ControlNet,
# no pose copy, no sequential CPU offload.
_inpaint_pipe = None
_inpaint_ckpt = None


def release_inpaint():
    global _inpaint_pipe, _inpaint_ckpt
    _inpaint_pipe = None
    _inpaint_ckpt = None
    _free_torch()


def _get_ip_inpaint(checkpoint_path):
    global _inpaint_pipe, _inpaint_ckpt
    if _inpaint_pipe is not None and _inpaint_ckpt == checkpoint_path:
        return _inpaint_pipe
    release_inpaint()
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    with models.no_auto_download():
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            checkpoint_path, torch_dtype=torch.float16)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    try:
        pipe.to("cuda")
    except Exception:
        pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _inpaint_pipe, _inpaint_ckpt = pipe, checkpoint_path
    return pipe


def head_excluded_body_mask(base_path, models_root, hair_up=2.0) -> "object":
    """White = body to inpaint; black = whole head (face+hair) + background, kept.
    Person via BiRefNet; head box = detected face dilated up/out, subtracted."""
    import os
    import numpy as np
    from PIL import Image
    from supremediffusion.models.segmentation import (segment_foreground,
                                                       release_segmentation_model)
    from . import deps
    deps.ensure({"kornia": "kornia"})  # BiRefNet modeling code
    mp = segment_foreground(base_path, models_root)
    release_segmentation_model()
    _free_torch()
    person = np.array(Image.open(mp).convert("L"))
    try:
        os.unlink(mp)
    except Exception:
        pass
    mask = (person > 127).astype("uint8") * 255
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(cv2.imread(base_path))
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            bw, bh = x2 - x1, y2 - y1
            H, W = mask.shape
            hx1, hx2 = max(0, int(x1 - bw * 0.6)), min(W, int(x2 + bw * 0.6))
            hy1, hy2 = max(0, int(y1 - bh * hair_up)), min(H, int(y2 + bh * 0.25))
            mask[hy1:hy2, hx1:hx2] = 0  # exclude the whole head from the inpaint region
    except Exception:
        logger.warning("head detection failed; inpainting the full person", exc_info=True)
    return Image.fromarray(mask, "L")


def ip_adapter_inpaint(checkpoint_path, target_path, reference_path, mask_image,
                       prompt, negative, denoise=0.7, ip_scale=0.8, steps=30,
                       cfg=6.0, seed=-1, out_dir=None, progress=None) -> str | None:
    """Apply a reference identity onto the masked region of a target via IP-Adapter
    inpaint. Returns the saved path."""
    import random as _random
    import time
    import torch
    from PIL import Image
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    pipe = _get_ip_inpaint(checkpoint_path)
    pipe.set_ip_adapter_scale(float(ip_scale))
    target = Image.open(target_path).convert("RGB")
    ref = Image.open(reference_path).convert("RGB")
    w, h = target.size
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    with models.no_auto_download():
        img = pipe(prompt=prompt or "", negative_prompt=negative or "",
                   image=target, mask_image=mask_image, ip_adapter_image=ref,
                   strength=float(denoise), num_inference_steps=int(steps),
                   guidance_scale=float(cfg), width=w, height=h, generator=gen).images[0]
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "swap")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"ipinpaint_{int(seed)}_{int(time.time())}.png"
    try:
        img.save(f)
        return str(f)
    except Exception:
        logger.warning("failed saving ip-adapter inpaint", exc_info=True)
        return None


def body_swap(checkpoint_path, base_path, source_person_path, prompt, negative,
              cn_strength=0.7, ip_scale=0.8, denoise=0.75, cfg=7.0, steps=30,
              seed=-1, sampler="DPM++ 3M SDE", scheduler="Karras",
              adetailer=True, progress=None) -> str | None:
    """Transfer the source person's skin tone / body texture onto the base's body
    (whole head excluded → face + hair preserved). No pose copy, no ControlNet —
    an IP-Adapter masked inpaint. SD-family checkpoints only."""
    def _say(frac, msg):
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except Exception:
                pass
    _import_pipeline_cls()
    models_root = str(Path(SD_CHECKOUT) / "models")
    release_sd()  # free the txt2img checkpoint
    _say(0.2, "Segmenting body (head excluded)…")
    mask = head_excluded_body_mask(base_path, models_root)
    _say(0.55, "Applying source skin/texture (IP-Adapter inpaint)…")
    return ip_adapter_inpaint(checkpoint_path, base_path, source_person_path, mask,
                              prompt, negative, denoise=float(denoise),
                              ip_scale=float(ip_scale), steps=int(steps),
                              cfg=float(cfg), seed=int(seed), progress=progress)
