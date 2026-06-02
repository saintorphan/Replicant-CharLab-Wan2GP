"""Character data model + on-disk format.

Ported from SupremeDiffusion's ``create_character.py`` with all Qt/app-state
coupling removed. This module is pure Python (dataclass + Pillow + pathlib) so it
can be unit-tested and reused independently of the Gradio UI.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("replicant.character")

# --- prompt framing / caption tags ----------------------------------------

BASE_FRAMING = (
    "solo, one person, full body photo, facing the viewer, standing front view, "
    "head to toe, entire body visible, wide shot"
)

# Distance + angle caption tags (trigger word leads, then these). Distance tags
# teach the model that blur belongs to wide shots, not the character.
DISTANCE_TAG = {"close": "close-up shot", "medium": "medium shot", "full": "full body shot"}
ANGLE_TAG = {
    "front": "front view", "three_quarter": "three-quarter view",
    "side": "side profile view", "back": "back view",
}

# 3-distance composition targets. Close-heavy for 512px video / low-VRAM (forces
# facial detail); balanced for high-res image LoRAs (full body still has detail).
RATIO_VIDEO = {"close": 0.60, "medium": 0.30, "full": 0.10}
RATIO_HIGHRES = {"close": 0.40, "medium": 0.30, "full": 0.30}


# Style → prompt "medium" word, used to seed the positive prompt before the
# Qwen3.5 enhancer expands it.
STYLE_MEDIUM = {"realism": "photo", "anime": "anime", "cartoon": "cartoon illustration"}

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, jpeg artifacts, blurry, deformed, "
    "disfigured, watermark, text, signature"
)


def build_seed_prompt(description: str, style: str = "realism") -> str:
    """Seed a positive prompt from the description + framing + style medium. This
    is the text the Qwen3.5 enhancer then expands into model-specific grammar."""
    medium = STYLE_MEDIUM.get(style, "photo")
    desc = (description or "").strip().rstrip(",")
    parts = [medium, BASE_FRAMING, desc]
    return ", ".join(p for p in parts if p)


# Individual framing clauses (from BASE_FRAMING) that lock the shot to a standing
# full-body front view. They belong on the BASE image, but must NOT ride along
# into per-pose prompts — otherwise every pose (sitting, kneeling, close-up, …) is
# dragged back to a standing full-body shot.
_FRAMING_CLAUSES = (
    "full body photo", "facing the viewer", "standing front view",
    "head to toe", "entire body visible", "wide shot", "full body shot",
)


def strip_base_framing(prompt: str) -> str:
    """Remove the base-image framing clauses from a prompt so it can be reused for
    pose generation (the per-pose description supplies the framing). Keeps the
    character description AND 'solo, one person' (the latter still helps suppress
    extra people in the pose shots). Comma-aware."""
    import re
    if not prompt:
        return prompt
    out = prompt
    for clause in _FRAMING_CLAUSES:  # NB: not "solo, one person" — that's kept
        out = re.sub(rf"(?i)\b{re.escape(clause)}\b", "", out)
    out = re.sub(r"\s*,\s*(?:,\s*)+", ", ", out)   # collapse empty comma slots
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip().strip(",").strip()


def caption_for(trigger: str, distance: str, angle: str, desc: str) -> str:
    """Trigger-first caption with distance + angle tags, then the description."""
    parts = [trigger, DISTANCE_TAG.get(distance, "full body shot"),
             ANGLE_TAG.get(angle, ""), desc]
    return ", ".join(p for p in parts if p).rstrip(", ")


def pose_filename(idx: int, distance: str, angle: str) -> str:
    """Encode distance + angle in the saved pose filename (``__`` separates
    fields so single-underscore values like ``three_quarter`` survive)."""
    return f"pose_{idx + 1:03d}__{distance}__{angle}.png"


def parse_pose_distance_angle(path) -> tuple[str, str]:
    """Recover (distance, angle) from a saved pose filename; legacy -> full/front."""
    parts = Path(path).stem.split("__")
    if len(parts) >= 3 and parts[1] in ("close", "medium", "full"):
        return parts[1], parts[2]
    return "full", "front"


def trigger_from_name(name: str) -> str:
    """Sanitize a character name into a LoRA trigger word."""
    safe = "".join(c if c.isalnum() else "_" for c in (name or "char")).strip("_").lower()
    return safe or "character"


@dataclass
class CharacterState:
    name: str = ""
    description: str = ""
    style: str = "realism"
    selected_loras: list = field(default_factory=list)
    lora_multipliers: str = ""
    lora_prompt_tags: str = ""      # "<lora:name:w>, ..." for prompt injection
    lora_trigger_words: str = ""    # "trigger1, trigger2, ..." from metadata
    positive_prompt: str = ""
    negative_prompt: str = ""
    reference_image: str = ""        # optional user-supplied character reference
    base_images: list = field(default_factory=list)
    selected_base: str = ""
    # Step 4 (Face/Body) records whether the user OPTIONALLY swapped the base
    # image. Pose generation always grabs the face from the finalized base.
    face_swap_enabled: bool = False
    face_source_path: str = ""
    face_result_path: str = ""
    face_enhancer: str = ""
    face_enhancer_strength: float = 0.5
    face_blend_ratio: float = 0.5
    body_swap_enabled: bool = False
    body_source_path: str = ""
    body_result_path: str = ""
    body_ip_scale: float = 0.8
    body_denoise: float = 0.75
    body_cfg: float = 7.0
    body_cn_strength: float = 0.7
    pose_images: list = field(default_factory=list)
    approved_poses: list = field(default_factory=list)
    # Parallel to approved_poses: {distance, angle, orientation, pose_index}.
    # Runtime-only; the durable source of truth on reload is the saved filename.
    approved_pose_specs: list = field(default_factory=list)
    checkpoint: str = ""
    model_family: str = ""           # native Wan2GP model type for generation
    sampler: str = ""
    scheduler: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    seed: int = -1
    ref_look_strength: float = 0.7   # IP-Adapter scale: canonical look -> poses
    apply_body_to_poses: bool = True
    adetailer: bool = True

    # Scalar fields persisted to character.json (image paths are saved as files
    # and reconstructed on load -- see load_character).
    META_FIELDS = (
        "name", "description", "style", "selected_loras", "lora_multipliers",
        "lora_prompt_tags", "lora_trigger_words", "positive_prompt", "negative_prompt",
        "face_swap_enabled", "face_enhancer", "face_enhancer_strength", "face_blend_ratio",
        "body_swap_enabled", "body_ip_scale", "body_denoise", "body_cfg", "body_cn_strength",
        "checkpoint", "model_family", "sampler", "scheduler", "steps", "cfg_scale",
        "width", "height", "seed", "ref_look_strength", "apply_body_to_poses", "adetailer",
    )

    @property
    def trigger(self) -> str:
        return trigger_from_name(self.name)

    def to_meta(self) -> dict:
        return {k: getattr(self, k) for k in self.META_FIELDS}

    def apply_meta(self, d: dict) -> None:
        for k in self.META_FIELDS:
            if k in d:
                setattr(self, k, d[k])


def load_character(char_dir) -> CharacterState:
    """Reconstruct a CharacterState from a saved character directory."""
    char_dir = Path(char_dir)
    cs = CharacterState()
    meta_path = char_dir / "character.json"
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                cs.apply_meta(json.load(f))
        except Exception:
            logger.warning("Failed to read %s", meta_path, exc_info=True)
    # Reconstruct image paths from the saved files.
    base = char_dir / "base.png"
    if base.is_file():
        cs.selected_base = str(base)
    ref = char_dir / "reference_ref.png"
    if ref.is_file():
        cs.reference_image = str(ref)
    face = char_dir / "face_ref.png"
    if face.is_file():
        cs.face_source_path = str(face)
    body = char_dir / "body_ref.png"
    if body.is_file():
        cs.body_source_path = str(body)
    poses_dir = char_dir / "poses"
    if poses_dir.is_dir():
        cs.approved_poses = sorted(str(p) for p in poses_dir.glob("*.png"))
        cs.pose_images = list(cs.approved_poses)
        cs.approved_pose_specs = [
            {"distance": d, "angle": a, "orientation":
                "landscape" if d == "full" and a == "side" else "portrait"}
            for d, a in (parse_pose_distance_angle(p) for p in cs.approved_poses)
        ]
    return cs


def save_character(char_dir, cs: CharacterState) -> Path:
    """Write character.json + copy the base / reference / swap-source / pose
    images into the character directory. Returns the directory path."""
    from PIL import Image
    char_dir = Path(char_dir)
    char_dir.mkdir(parents=True, exist_ok=True)

    with open(char_dir / "character.json", "w") as f:
        json.dump(cs.to_meta(), f, indent=2)

    def _copy(src: str, name: str) -> None:
        if src and Path(src).is_file():
            try:
                Image.open(src).convert("RGB").save(char_dir / name)
            except Exception:
                logger.warning("Failed to save %s", name, exc_info=True)

    _copy(cs.selected_base, "base.png")
    _copy(cs.reference_image, "reference_ref.png")
    _copy(cs.face_source_path, "face_ref.png")
    _copy(cs.body_source_path, "body_ref.png")

    poses_dir = char_dir / "poses"
    poses_dir.mkdir(exist_ok=True)
    for old in poses_dir.glob("*.png"):
        old.unlink()
    specs = cs.approved_pose_specs or []
    for i, pose in enumerate(cs.approved_poses):
        if not (pose and Path(pose).is_file()):
            continue
        sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
        dst = poses_dir / pose_filename(i, sp.get("distance", "full"), sp.get("angle", "front"))
        try:
            Image.open(pose).convert("RGB").save(dst)
        except Exception:
            logger.warning("Failed to save pose %s", pose, exc_info=True)
    return char_dir
