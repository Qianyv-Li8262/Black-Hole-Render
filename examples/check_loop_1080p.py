# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Check the latest 1080p example's start, midpoint and endpoint PNGs."""
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    folder = Path(__file__).resolve().parents[1] / 'output_frames/repro_1080p/loop'
    manifest_path = max((path for path in folder.glob('run_single_gpu_*.json')
                         if not path.stem.endswith('_camera')), default=None)
    if manifest_path is None:
        raise SystemExit(f'No A/B render manifest in {folder}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    loop = manifest['disk_loop']
    if (manifest['status'] != 'complete' or manifest['frames_rendered'] != 3
            or manifest['camera_motion'] != 'static' or not loop['enabled']):
        raise SystemExit('The latest render must contain three frames with a static camera and A/B enabled.')

    trajectory = json.loads((folder / manifest['camera_trajectory']).read_text(encoding='utf-8'))
    times = np.array([frame['time'] for frame in trajectory])
    if len(times) != 3 or not np.allclose(np.diff(times), loop['period'] / 2, rtol=0, atol=1e-7):
        raise SystemExit('The camera trajectory must sample t, t+T/2 and t+T.')

    images = []
    for frame in trajectory:
        path = folder / f"frame_{frame['frame']:05d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (1080, 1920, 3):
            raise SystemExit(f'Expected a 1920×1080 RGB frame at {path}')
        images.append(image.astype(np.int16))

    start, midpoint, end = images
    endpoint_difference = np.abs(start - end)
    endpoint_max = int(endpoint_difference.max())
    endpoint_mae = float(endpoint_difference.mean())
    midpoint_mae = float(np.abs(start - midpoint).mean())
    print(json.dumps({
        'coordinate_times': times.tolist(),
        'difference_units': '8-bit RGB levels',
        'endpoint_max': endpoint_max,
        'endpoint_mae': endpoint_mae,
        'midpoint_mae': midpoint_mae,
    }, indent=2))
    if endpoint_max > 2 or endpoint_mae > 0.01 or midpoint_mae < 0.05:
        raise SystemExit('Loop check failed: endpoints must match and the midpoint must change.')
    print('PASS: the midpoint changes and the final frame closes the loop.')


if __name__ == '__main__':
    main()
