"""Auto-install missing runtime *code* dependencies so features self-heal instead
of hard-failing mid-run. (Models are never auto-pulled — that's core.models.)"""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys

logger = logging.getLogger("replicant.deps")

# import-name -> pip spec. The body-swap / SD-image path needs these beyond what
# Wan2GP bundles.
BODY_SWAP_DEPS = {
    "kornia": "kornia",              # BiRefNet segmentation custom modeling code
    "controlnet_aux": "controlnet_aux",  # OpenPose preprocessor
    "ultralytics": "ultralytics",    # YOLOv8 (ADetailer / person detection)
}


def ensure(import_to_pip: dict) -> None:
    """pip-install any of import_to_pip whose import-name isn't importable."""
    missing = []
    for imp, pip in import_to_pip.items():
        try:
            importlib.import_module(imp)
        except Exception:
            missing.append(pip)
    if not missing:
        return
    logger.info("Replicant: auto-installing missing deps: %s", missing)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception:
        logger.warning("auto-install failed for %s", missing, exc_info=True)


def ensure_body_swap() -> None:
    ensure(BODY_SWAP_DEPS)
