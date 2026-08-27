"""Small helper for saving the settings used by an offline render."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import platform
from types import ModuleType

import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError


def snapshot_settings(source) -> dict:
    """Return JSON-friendly public values from a module or mapping."""

    values = vars(source) if isinstance(source, ModuleType) else source
    result = {}
    for name, value in values.items():
        if name.startswith("_") or callable(value) or isinstance(value, ModuleType):
            continue
        try:
            result[name] = _json_value(value)
        except TypeError:
            pass
    return result


def start_manifest(output_dir, renderer: str, settings, extra=None) -> Path:
    output_dir = Path(output_dir)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"run_{renderer}_{stamp}.json"
    data = {
        "renderer": renderer,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "settings": snapshot_settings(settings),
    }
    if extra:
        data.update(_json_value(extra))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def finish_manifest(path: Path, status: str, extra=None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    data["finished_at"] = datetime.now().astimezone().isoformat()
    if extra:
        data.update(_json_value(extra))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
