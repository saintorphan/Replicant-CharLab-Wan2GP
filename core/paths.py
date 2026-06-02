"""Filesystem layout for Replicant Character Lab.

Two kinds of location:

  * Plugin-specific — under ``<wan2gp_root>/character_lab/`` (override
    REPLICANT_LAB_DIR), persisted to ``<wan2gp_root>/.replicant_charlab.json``:
        characters_dir   saved characters
        datasets_dir     per-character LoRA training datasets

  * OrphanSuite SHARED — model resources several saintorphan plugins use, kept in
    ONE central place ``<wan2gp_root>/orphansuite/`` (override ORPHANSUITE_DIR) so
    they aren't duplicated per plugin, with the paths stored in the cross-plugin
    ``<wan2gp_root>/.orphansuite.json``:
        models_dir       face / ADetailer / face-swap / BiRefNet weights (download target)
        sdxl_models_dir  SDXL/Pony/Illustrious checkpoints
        sdxl_loras_dir   SDXL-family LoRAs

Set a shared dir once (Settings → OrphanSuite) and every plugin (Image Suite,
Replicant CharLab, Reel2Reel) reads the same value. Shared dirs fall back to the
legacy ``character_lab/<leaf>`` location for back-compat, and the Settings "link
existing folder" action symlinks models you already have (a1111/Forge/…) into the
shared area without copying.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("replicant.paths")

_DEFAULT_SUBDIR = "character_lab"
_CONFIG_NAME = ".replicant_charlab.json"
_config: dict | None = None


# --- config persistence (plugin-specific: characters / datasets) -----------

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
    """Persist directory overrides. Plugin-specific dirs (characters, datasets) go
    to .replicant_charlab.json; SHARED model resources (download target, SDXL
    checkpoints + LoRAs) go to the cross-plugin .orphansuite.json so every plugin
    follows. '' clears an override (reverts to the default)."""
    cfg = load_config()
    for key, val in (("characters_dir", characters), ("datasets_dir", datasets)):
        if val is not None:
            cfg[key] = str(Path(val).expanduser()) if val else ""
    save_config()
    for key, val in (("models_dir", models), ("sdxl_models_dir", sdxl_models),
                     ("sdxl_loras_dir", sdxl_loras)):
        if val is not None:
            set_shared_dir(key, val)
    ensure_dirs()


# --- roots -----------------------------------------------------------------

def lab_root() -> Path:
    override = os.environ.get("REPLICANT_LAB_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _DEFAULT_SUBDIR


def _dir(key: str, default_leaf: str) -> Path:
    val = load_config().get(key)
    return Path(val).expanduser() if val else lab_root() / default_leaf


# --- OrphanSuite shared resources ------------------------------------------
# SDXL/Pony/Illustrious checkpoints, SDXL LoRAs and face/ADetailer/face-swap/
# BiRefNet weights are SHARED across the saintorphan plugins, so their paths live
# in ONE central config — ``.orphansuite.json`` at the Wan2GP root, NOT each
# plugin's own config. Override the shared root directory with ORPHANSUITE_DIR.
#
# Resolution for a shared dir (first hit wins):
#   1. .orphansuite.json[key]          — the shared, canonical cross-plugin setting
#   2. .replicant_charlab.json[key]    — legacy per-plugin override
#   3. <root>/orphansuite/<leaf>       — the shared area, if it already holds files
#   4. <lab_root>/<leaf>               — legacy CharLab location (back-compat)
#   5. <root>/orphansuite/<leaf>       — default (created/symlinked on demand)
_ORPHAN_CONFIG_NAME = ".orphansuite.json"
_ORPHAN_SUBDIR = "orphansuite"
_orphan_cfg: dict | None = None
_orphan_mtime = None  # cache key — reload when the file changes on disk


def orphansuite_root() -> Path:
    """Central shared-resource root for all saintorphan plugins."""
    override = os.environ.get("ORPHANSUITE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _ORPHAN_SUBDIR


def _orphan_config_path() -> Path:
    return Path(os.getcwd()) / _ORPHAN_CONFIG_NAME


def _read_orphan_disk() -> dict:
    try:
        d = json.loads(_orphan_config_path().read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def load_shared_config() -> dict:
    """The cross-plugin .orphansuite.json (shared model/dir settings). Reloads when
    the file changes on disk so a sibling plugin's writes aren't stale (the file is
    tiny — this is cheap)."""
    global _orphan_cfg, _orphan_mtime
    try:
        mt = _orphan_config_path().stat().st_mtime
    except Exception:
        mt = None
    if _orphan_cfg is None or mt != _orphan_mtime:
        _orphan_cfg = _read_orphan_disk()
        _orphan_mtime = mt
    return _orphan_cfg


def get_shared(key: str, default=None):
    """Read any value from the cross-plugin .orphansuite.json."""
    v = load_shared_config().get(key)
    return default if v is None else v


def set_shared(key: str, value) -> None:
    """Persist any JSON value to the cross-plugin .orphansuite.json. Re-reads the
    file fresh and MERGES, so a concurrent sibling plugin's keys aren't clobbered."""
    global _orphan_cfg, _orphan_mtime
    cfg = _read_orphan_disk()  # fresh from disk, not the (possibly stale) cache
    cfg[key] = value
    try:
        _orphan_config_path().write_text(json.dumps(cfg, indent=2))
    except Exception:
        logger.warning("could not write %s", _ORPHAN_CONFIG_NAME, exc_info=True)
    _orphan_cfg = cfg
    try:
        _orphan_mtime = _orphan_config_path().stat().st_mtime
    except Exception:
        _orphan_mtime = None


def set_shared_dir(key: str, path: str) -> None:
    """Persist a shared resource dir to the cross-plugin .orphansuite.json."""
    set_shared(key, str(Path(path).expanduser()) if path else "")


def _has_files(d: Path) -> bool:
    try:
        return d.is_dir() and any(d.iterdir())
    except Exception:
        return False


def _shared_dir(key: str, default_leaf: str) -> Path:
    """Resolve a shared (cross-plugin) resource dir — see resolution order above."""
    sv = load_shared_config().get(key)
    if sv:
        return Path(sv).expanduser()
    own = load_config().get(key)            # legacy per-plugin override
    if own:
        return Path(own).expanduser()
    shared = orphansuite_root() / default_leaf
    legacy = lab_root() / default_leaf      # where CharLab kept them before
    if _has_files(shared):
        return shared
    if _has_files(legacy):
        return legacy
    return shared


# Logical link targets → the dir consumers actually read. Resolved through the
# SAME functions the loaders use (so a configured custom dir is honoured), and the
# face/body/birefnet weights land in the partitioned subdirs the loaders expect.
LINK_TARGETS = ["sdxl_models", "sdxl_loras", "face", "body", "birefnet"]


def link_target_dir(target: str) -> Path:
    """Resolve a logical link target to the exact dir its loader scans."""
    if target == "sdxl_models":
        return sdxl_models_dir()
    if target == "sdxl_loras":
        return sdxl_loras_dir()
    if target in ("face", "body", "birefnet"):
        return models_dir() / target  # loaders read models_dir()/<face|body|birefnet>
    return models_dir() / target


def link_existing_into_shared(
        source_dir: str, target: str,
        exts=(".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin",
              ".onnx", ".sft", ".vae")) -> str:
    """Symlink model files from ``source_dir`` into the dir its loader actually
    scans (``link_target_dir`` — honours a configured custom dir AND the
    face/body/birefnet subdir layout), so models you already keep (a1111 / Forge /
    a drive folder) are reused without copying or moving the originals. Real
    symlinks on POSIX; copies as a fallback on Windows where symlinks need admin.
    Existing symlinks in source are followed to their real target. Returns a summary."""
    import shutil
    src = Path(source_dir).expanduser()
    if not src.is_dir():
        raise ValueError(f"Not a folder: {source_dir}")
    dst_root = link_target_dir(target)
    dst_root.mkdir(parents=True, exist_ok=True)
    linked = copied = skipped = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or (exts and f.suffix.lower() not in exts):
            continue
        dst = dst_root / f.name
        if dst.exists() or dst.is_symlink():
            skipped += 1
            continue
        target = f.resolve()  # follow existing symlinks to the real file
        try:
            dst.symlink_to(target)
            linked += 1
        except OSError:
            try:
                shutil.copy2(target, dst)
                copied += 1
            except Exception:
                logger.warning("could not link/copy %s", f, exc_info=True)
    bits = []
    if linked:
        bits.append(f"linked {linked}")
    if copied:
        bits.append(f"copied {copied}")
    if skipped:
        bits.append(f"skipped {skipped} (already there)")
    return (", ".join(bits) or "no model files found") + f" → {dst_root}"


# --- the dirs --------------------------------------------------------------

def characters_dir() -> Path:
    return _dir("characters_dir", "characters")


def datasets_dir() -> Path:
    return _dir("datasets_dir", "datasets")


def models_dir() -> Path:
    """Download target for face / ADetailer / face-swap / BiRefNet weights (shared)."""
    return _shared_dir("models_dir", "models")


def sdxl_models_dir() -> Path:
    """SDXL/Pony/Illustrious checkpoints (shared across plugins)."""
    return _shared_dir("sdxl_models_dir", "sdxl_models")


def sdxl_loras_dir() -> Path:
    """SDXL-family LoRAs (shared across plugins)."""
    return _shared_dir("sdxl_loras_dir", "sdxl_loras")


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
    for d in (characters_dir(), datasets_dir(), models_dir(), cache_dir(),
              sdxl_models_dir(), sdxl_loras_dir()):
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
