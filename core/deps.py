"""Auto-install missing runtime *code* dependencies so features self-heal instead
of hard-failing mid-run. (Models are never auto-pulled — that's core.models.)"""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys

logger = logging.getLogger("replicant.deps")

# import-name -> pip spec. The body-swap / SD-image path needs these beyond what
# Wan2GP bundles. (controlnet_aux is intentionally NOT here — Replicant's body swap
# is an IP-Adapter inpaint with no ControlNet, so it's never imported.)
BODY_SWAP_DEPS = {
    "kornia": "kornia",              # BiRefNet segmentation custom modeling code
    "ultralytics": "ultralytics",    # YOLOv8 (ADetailer / person detection)
}


def _importable(imp: str) -> bool:
    try:
        importlib.import_module(imp)
        return True
    except Exception:
        return False


def ensure(import_to_pip: dict, progress=None, label="dependencies") -> None:
    """pip-install any of import_to_pip whose import-name isn't importable, then
    RE-VERIFY. Raises RuntimeError (not a silent fallback) if install fails or the
    module is still unimportable, so callers surface a clear message instead of a
    cryptic ModuleNotFoundError deep in generation. ``progress`` shows a status."""
    missing = {imp: pip for imp, pip in import_to_pip.items() if not _importable(imp)}
    if not missing:
        return
    pips = list(missing.values())
    logger.info("Replicant: auto-installing missing deps: %s", pips)
    if progress is not None:
        try:
            progress(0.0, desc=f"Installing {label}: {', '.join(pips)} "
                               f"(first run only — see console for progress)…")
        except Exception:
            pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pips])
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-install {label} ({', '.join(pips)}): {e}. "
            f"Install manually: pip install {' '.join(pips)}") from e
    importlib.invalidate_caches()  # make freshly-installed modules importable now
    still = [pip for imp, pip in missing.items() if not _importable(imp)]
    if still:
        raise RuntimeError(
            f"{label} installed but still not importable: {', '.join(still)}. "
            f"Restart the app, or install manually: pip install {' '.join(still)}")


def ensure_body_swap(progress=None) -> None:
    ensure(BODY_SWAP_DEPS, progress=progress, label="body-swap dependencies")
