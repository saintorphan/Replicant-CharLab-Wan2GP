"""LoRA dataset assembly.

Ported from SupremeDiffusion's ``_build_character_datasets`` / ``_compose_dataset``.
Builds per-distance pools from the approved poses (plus head / upper-body crops),
then composes the 3-distance datasets the trainers consume:

    video512  (close 60 / medium 30 / full 10)  -- 512px video + low-VRAM LoRA
    highres   (close 40 / medium 30 / full 30)  -- high-res image LoRA
    full      (full body only)                  -- back-compat
    face      (close crops only)                -- back-compat

The face/upper-body crops need a face detector exposing ``detect_faces(path) ->
[{"bbox": (x1, y1, x2, y2)}]``. SupremeDiffusion's FaceSwapPipeline satisfies this;
``default_face_detector()`` provides an InsightFace-backed fallback.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from .character import (CharacterState, RATIO_HIGHRES, RATIO_VIDEO, caption_for)

logger = logging.getLogger("replicant.datasets")


def default_face_detector():
    """InsightFace-backed detector with a ``detect_faces`` method. Returns None if
    InsightFace is unavailable (crops are then skipped)."""
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        logger.warning("InsightFace unavailable -- dataset face/body crops skipped")
        return None

    class _Detector:
        def __init__(self):
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0)

        def detect_faces(self, image_path):
            import numpy as np
            from PIL import Image
            img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]  # RGB->BGR
            return [{"bbox": tuple(map(int, f.bbox))} for f in self.app.get(img)]

    try:
        return _Detector()
    except Exception:
        logger.warning("InsightFace init failed -- dataset crops skipped", exc_info=True)
        return None


def _largest_face_bbox(detector, image_path):
    faces = detector.detect_faces(image_path)
    if not faces:
        return None
    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
               reverse=True)
    return faces[0]["bbox"]


def crop_head(image_path: str, out_path: str, detector) -> bool:
    """Crop the head/face region (with padding for hair) from an image."""
    from PIL import Image
    try:
        bbox = _largest_face_bbox(detector, image_path)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * 0.6), int(bh * 0.6)
        left = max(0, x1 - px)
        top = max(0, y1 - int(py * 1.3))
        right = min(W, x2 + px)
        bottom = min(H, y2 + py)
        img.crop((left, top, right, bottom)).save(out_path)
        return True
    except Exception:
        logger.debug("head crop failed for %s", image_path, exc_info=True)
        return False


def crop_upper_body(image_path: str, out_path: str, detector) -> bool:
    """Crop a waist-up (medium) region using the detected face as an anchor."""
    from PIL import Image
    try:
        bbox = _largest_face_bbox(detector, image_path)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        top = max(0, int(y1 - 0.6 * bh))
        bottom = min(H, int(y2 + 4.5 * bh))
        half_w = 1.6 * bw
        left = max(0, int(cx - half_w))
        right = min(W, int(cx + half_w))
        if right - left < 16 or bottom - top < 16:
            return False
        img.crop((left, top, right, bottom)).save(out_path)
        return True
    except Exception:
        logger.debug("upper-body crop failed for %s", image_path, exc_info=True)
        return False


def _save_resized(src, dst, max_side: int | None):
    """Save src->dst, downscaling so the long edge <= max_side (Lanczos, aspect kept).
    Downsampling from the ~832x1216 generation supersamples away diffusion grain and
    anti-aliases edges, so the training crop is crisper than a native-512 render."""
    from PIL import Image
    img = Image.open(src).convert("RGB")
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                             Image.Resampling.LANCZOS)
    img.save(dst)


def compose_dataset(pools: dict, out_dir: Path, ratio: dict, *, total: int | None = None,
                    max_side: int | None = None) -> str:
    """Assemble a FLAT folder of NNN.png + NNN.txt sampled from per-distance pools
    to hit ``ratio``. Oversamples (duplicates) a short pool, subsets a long one.
    ``max_side`` downscales the long edge (Lanczos) — e.g. 512 for the training set."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()
    avail = {k: list(v) for k, v in pools.items() if k in ratio}
    if not any(avail.values()):
        return str(out_dir)
    if total is None:
        total = sum(len(v) for v in avail.values())
    rng = random.Random(42)
    n = 0
    for dist, frac in ratio.items():
        pool = avail.get(dist) or []
        if not pool:
            continue
        want = max(1, round(frac * total))
        if len(pool) >= want:
            chosen = rng.sample(pool, want)
        else:  # oversample with duplication
            chosen = list(pool) + [rng.choice(pool) for _ in range(want - len(pool))]
            logger.info("compose: %s pool short (%d/%d) -- duplicated to balance",
                        dist, len(pool), want)
        for img, cap in chosen:
            n += 1
            dst = out_dir / f"{n:03d}.png"
            try:
                _save_resized(img, dst, max_side)
            except Exception:
                import shutil
                shutil.copy2(img, dst)
            dst.with_suffix(".txt").write_text(cap)
    return str(out_dir)


def build_character_datasets(char_dir, cs: CharacterState, detector=None) -> dict:
    """Build per-distance pools from approved poses (+ head/upper-body crops), then
    compose the 3-distance datasets. Pass a face ``detector`` (defaults to
    InsightFace). Returns {"trigger", "video512", "highres", "full", "face"}."""
    import shutil
    from PIL import Image

    char_dir = Path(char_dir)
    trigger = cs.trigger
    desc = (cs.description or cs.positive_prompt or "").strip()
    poses = [p for p in cs.approved_poses if p and Path(p).is_file()]
    specs = cs.approved_pose_specs or []

    pool_root = char_dir / "datasets" / "_pool"
    for sub in ("close", "medium", "full"):
        (pool_root / sub).mkdir(parents=True, exist_ok=True)
        for old in (pool_root / sub).glob("*"):
            old.unlink()
    pools: dict = {"close": [], "medium": [], "full": []}

    def _add(dist: str, src_img: str, angle: str) -> None:
        idx = len(pools[dist]) + 1
        dst = pool_root / dist / f"{idx:03d}.png"
        try:
            Image.open(src_img).convert("RGB").save(dst)
        except Exception:
            shutil.copy2(src_img, dst)
        pools[dist].append((str(dst), caption_for(trigger, dist, angle, desc)))

    # 1) Each generated pose -> its own distance pool.
    for i, pose in enumerate(poses):
        sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
        _add(sp.get("distance", "full"), pose, sp.get("angle", "front"))

    # 2) Crop close (head) + medium (waist-up) from full/medium poses.
    if detector is None:
        detector = default_face_detector()
    if detector is not None:
        try:
            crop_dir = pool_root / "_crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            for old in crop_dir.glob("*"):
                old.unlink()
            cn = 0
            for i, pose in enumerate(poses):
                sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
                dist, angle = sp.get("distance", "full"), sp.get("angle", "front")
                if dist in ("full", "medium"):
                    cn += 1
                    head = crop_dir / f"head_{cn:03d}.png"
                    if crop_head(pose, str(head), detector):
                        _add("close", str(head), angle)
                if dist == "full":
                    cn += 1
                    up = crop_dir / f"upper_{cn:03d}.png"
                    if crop_upper_body(pose, str(up), detector):
                        _add("medium", str(up), angle)
        except Exception:
            logger.warning("Crop pool build failed", exc_info=True)

    datasets_root = char_dir / "datasets"
    # Training cap is 512: downscale video512 to a 512 long edge (Lanczos). highres
    # keeps a larger 768 long edge; full/face stay native.
    video512 = compose_dataset(pools, datasets_root / "video512", RATIO_VIDEO, max_side=512)
    highres = compose_dataset(pools, datasets_root / "highres", RATIO_HIGHRES, max_side=768)
    full_only = compose_dataset({"full": pools["full"]}, datasets_root / "full", {"full": 1.0})
    face_only = (compose_dataset({"close": pools["close"]}, datasets_root / "face", {"close": 1.0})
                 if pools["close"] else None)

    logger.info("datasets: pools close=%d medium=%d full=%d", len(pools["close"]),
                len(pools["medium"]), len(pools["full"]))
    return {"trigger": trigger, "video512": video512, "highres": highres,
            "full": full_only, "face": face_only}
