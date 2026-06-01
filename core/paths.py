"""Filesystem layout for Replicant Character Lab.

Three roots, each independently overridable from the wizard's Directories panel
and persisted to ``<wan2gp_root>/.replicant_charlab.json`` so the choice survives
restarts:

    characters_dir   default <wan2gp_root>/character_lab/characters
    datasets_dir     default <wan2gp_root>/character_lab/datasets   (one sub-dir per character)
    models_dir       default <wan2gp_root>/character_lab/models     (download buttons stash here)

REPLICANT_LAB_DIR overrides the default root for all three at once.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("replicant.paths")

_DEFAULT_SUBDIR = "character_lab"
_CONFIG_NAME = ".replicant_charlab.json"
_KEYS = ("characters_dir", "datasets_dir", "models_dir")
_config: dict | None = None


# --- config persistence ----------------------------------------------------

def _config_path() -> Path:
    # Stable location (cwd = Wan2GP root), independent of the configurable dirs.
    return Path(os.getcwd()) / _CONFIG_NAME


def load_config() -> dict:
    global _config
    if _config is None:
        try:
            _config = json.loads(_config_path().read_text())
        except Exception:
            _config = {}
    return _config


def save_config() -> None:
    try:
        _config_path().write_text(json.dumps(load_config(), indent=2))
    except Exception:
        logger.warning("Could not write %s", _config_path(), exc_info=True)


def set_dirs(*, characters=None, datasets=None, models=None,
             sdxl_models=None, sdxl_loras=None) -> None:
    """Override any of the roots (absolute paths) and persist. SDXL model/LoRA
    dirs are optional external paths (e.g. an a1111/forge install) — stored only,
    never created."""
    cfg = load_config()
    for key, val in (("characters_dir", characters), ("datasets_dir", datasets),
                     ("models_dir", models), ("sdxl_models_dir", sdxl_models),
                     ("sdxl_loras_dir", sdxl_loras)):
        if val is not None:
            cfg[key] = str(Path(val).expanduser()) if val else ""
    save_config()
    ensure_dirs()


def sdxl_models_dir() -> str:
    """Optional external dir of SDXL/Pony/Illustrious checkpoints ('' if unset)."""
    return load_config().get("sdxl_models_dir", "")


def sdxl_loras_dir() -> str:
    """Optional external dir of SDXL-family LoRAs ('' if unset)."""
    return load_config().get("sdxl_loras_dir", "")


# --- roots -----------------------------------------------------------------

def lab_root() -> Path:
    override = os.environ.get("REPLICANT_LAB_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _DEFAULT_SUBDIR


def _dir(key: str, default_leaf: str) -> Path:
    val = load_config().get(key)
    return Path(val).expanduser() if val else lab_root() / default_leaf


def characters_dir() -> Path:
    return _dir("characters_dir", "characters")


def datasets_dir() -> Path:
    return _dir("datasets_dir", "datasets")


def models_dir() -> Path:
    return _dir("models_dir", "models")


def cache_dir() -> Path:
    return lab_root() / ".cache"


def _safe(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in (name or "")).strip()
    return s or "unnamed"


def character_dir(name: str) -> Path:
    return characters_dir() / _safe(name)


def character_dataset_dir(name: str) -> Path:
    return datasets_dir() / _safe(name)


def ensure_dirs() -> Path:
    """Create the directory tree if missing. Idempotent; called on plugin setup."""
    for d in (characters_dir(), datasets_dir(), models_dir(), cache_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning("Could not create %s", d, exc_info=True)
    return lab_root()


def list_characters() -> list[str]:
    cdir = characters_dir()
    if not cdir.is_dir():
        return []
    return sorted(p.name for p in cdir.iterdir()
                  if p.is_dir() and (p / "character.json").is_file())
