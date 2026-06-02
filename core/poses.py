"""Predefined pose set for step 5, ported from SupremeDiffusion.

The generation set is weighted toward FULL + MEDIUM with varied angles; the
close-up majority the 512px dataset formula wants is manufactured later by
cropping heads/upper-bodies from these (see core.datasets).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseSpec:
    description: str
    distance: str                    # "close" | "medium" | "full"
    angle: str                       # "front" | "three_quarter" | "side" | "back"
    orientation: str = "portrait"    # "portrait" | "square" (sitting) | "landscape"


POSES: list[PoseSpec] = [
    # Full body — angle coverage (front / three-quarter / side / back)
    PoseSpec("full body shot, standing front view, arms at sides, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, standing three-quarter view, slight turn, feet visible, head to toe, wide shot", "full", "three_quarter"),
    PoseSpec("full body shot, standing side profile view, feet visible, head to toe, wide shot", "full", "side"),
    PoseSpec("full body shot, standing back view from behind, feet visible, head to toe, wide shot", "full", "back"),
    PoseSpec("full body shot, standing, hands on hips, confident, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, walking forward dynamic stride, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, sitting in a chair front view, hands on lap, legs visible, wide framing", "full", "front", "square"),
    PoseSpec("full body shot, sitting in a chair three-quarter view, legs crossed, whole body visible, wide framing", "full", "three_quarter", "square"),
    PoseSpec("full body shot, sitting cross-legged on the ground, casual pose, whole body visible, wide framing", "full", "front", "square"),
    PoseSpec("full body shot, kneeling, upright posture, whole body visible, wide framing", "full", "three_quarter", "square"),
    PoseSpec("full body shot, reclining on a sofa, leaning back, one arm resting, entire body in frame", "full", "side", "landscape"),
    PoseSpec("full body shot, laying on side, head resting on arm, peaceful, entire body in frame", "full", "side", "landscape"),
    # Medium / waist-up — varied angles
    PoseSpec("medium shot, waist up, front view, relaxed expression, neutral background", "medium", "front"),
    PoseSpec("medium shot, waist up, three-quarter view, soft smile, neutral background", "medium", "three_quarter"),
    PoseSpec("medium shot, waist up, side profile view, neutral background", "medium", "side"),
    PoseSpec("medium shot, upper body, hands on hips, front view, neutral background", "medium", "front"),
    PoseSpec("medium shot, upper body, arms crossed, three-quarter view, neutral background", "medium", "three_quarter"),
    # Close-up portraits (a few genuine ones; the rest are cropped from full/medium)
    PoseSpec("close-up portrait, head and shoulders, front view, neutral expression, sharp focus on face", "close", "front"),
    PoseSpec("close-up portrait, head and shoulders, three-quarter view, gentle smile, sharp focus on face", "close", "three_quarter"),
    PoseSpec("close-up portrait, head and shoulders, side profile view, sharp focus on face", "close", "side"),
    PoseSpec("close-up portrait, head and shoulders, front view, soft smile, sharp focus on face", "close", "front"),
]

# Appended to the user negative during pose gen to prevent close-up/portrait
# framing leaking into full-body shots.
POSE_NEGATIVE_EXTRA = (
    "close-up, closeup, portrait, headshot, cropped, upper body only, "
    "face only, bust shot, shoulders up, head and shoulders, "
    "cut off, out of frame, partial body"
)


def pose_negative_for(distance: str, base_neg: str) -> str:
    """Distance-conditional negative. Full keeps the anti-crop bans; medium and
    close MUST NOT ban close-ups or they never frame tight."""
    base = (base_neg or "").strip().rstrip(",")
    if distance == "full":
        extra = POSE_NEGATIVE_EXTRA
    elif distance == "medium":
        extra = "cut off, out of frame, partial body, deformed, extra limbs"
    else:  # close
        extra = "full body, wide shot, cropped face, out of frame, deformed"
    return f"{base}, {extra}" if base else extra
