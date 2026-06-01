"""Filesystem layout for Replicant Character Lab.

On first run the plugin creates its data directories under the Wan2GP root:

    <wan2gp_root>/character_lab/
        characters/        # one sub-dir per saved character (character.json, base.png, poses/, datasets/)
        loras/             # trained character LoRAs land here
        .cache/            # scratch (crops, candidate gens) -- safe to wipe

The root can be overridden with the REPLICANT_LAB_DIR environment variable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("replicant.paths")

_DEFAULT_SUBDIR = "character_lab"


def lab_root() -> Path:
    """Resolve the Character Lab root. Honors REPLICANT_LAB_DIR; otherwise sits
    under the Wan2GP working directory (cwd when wgp.py runs)."""
    override = os.environ.get("REPLICANT_LAB_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _DEFAULT_SUBDIR


def characters_dir() -> Path:
    return lab_root() / "characters"


def loras_dir() -> Path:
    return lab_root() / "loras"


def cache_dir() -> Path:
    return lab_root() / ".cache"


def character_dir(name: str) -> Path:
    """Directory for a single named character (sanitized)."""
    safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in (name or "")).strip()
    return characters_dir() / (safe or "unnamed")


def ensure_dirs() -> Path:
    """Create the Character Lab directory tree if missing. Idempotent; called on
    plugin setup (first install creates it, later runs are no-ops). Returns root."""
    root = lab_root()
    for d in (characters_dir(), loras_dir(), cache_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning("Could not create %s", d, exc_info=True)
    logger.info("Replicant Character Lab root: %s", root)
    return root


def list_characters() -> list[str]:
    """Names of saved characters (dirs containing a character.json)."""
    cdir = characters_dir()
    if not cdir.is_dir():
        return []
    return sorted(p.name for p in cdir.iterdir()
                  if p.is_dir() and (p / "character.json").is_file())
