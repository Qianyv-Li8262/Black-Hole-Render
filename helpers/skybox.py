# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Load a linear RGB OpenEXR skybox in OpenCV's BGR layout."""
from pathlib import Path

import numpy as np


def read_skybox_bgr(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if path.suffix.lower() != '.exr':
        raise ValueError(f'Skybox must be an OpenEXR (.exr) image: {path}')
    if not path.is_file():
        return None

    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as exr:
        channels = exr.channels()
        missing = [name for name in ('R', 'G', 'B') if name not in channels]
        if missing:
            raise ValueError(f'Skybox {path} lacks OpenEXR channels {missing}')
        return np.stack((channels['B'].pixels, channels['G'].pixels, channels['R'].pixels),
                        axis=-1).astype(np.float32, copy=False)
