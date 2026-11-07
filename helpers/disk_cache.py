# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Validate disk texture semantics before selecting a rendering kernel."""
import json
from pathlib import Path
import numpy as np


def disk_half_extent(path):
    """Physical vertical texture bound; old caches retain their 2.5M domain."""
    sidecar = Path(path).with_suffix('.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    extent = float(metadata.get('vertical_half_extent', 2.5))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Disk vertical_half_extent must be positive and finite')
    return extent


def load_disk_cache(path, coefficients=False):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    encoded = metadata.get('format') == 'native34-coefficients-v1'
    if encoded != bool(coefficients):
        raise ValueError('Disk cache format and DISK_COEFFICIENTS disagree; generate cache with tools/prebake_native34.py')
    result = np.load(path)
    if result.ndim != 4 or result.shape[-1] != 4:
        raise ValueError('Invalid disk texture shape')
    if encoded and result.dtype != np.float32:
        raise ValueError('Independent disk coefficients must be stored in float32')
    return result
