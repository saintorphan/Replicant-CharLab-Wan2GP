"""Plugin-local persistence of the wizard's field values.

Autosaved to ``<lab_root>/.wizard_state.json`` on every field change and restored
on launch, so an app closure (or browser refresh) doesn't lose your inputs.
"""
from __future__ import annotations

import json
import logging

from . import paths

logger = logging.getLogger("replicant.wizard_state")


def _path():
    return paths.lab_root() / ".wizard_state.json"


def load() -> dict:
    try:
        return json.loads(_path().read_text())
    except Exception:
        return {}


def save(data: dict) -> None:
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        logger.debug("wizard state save failed", exc_info=True)


def clear() -> None:
    try:
        _path().unlink(missing_ok=True)
    except Exception:
        pass
