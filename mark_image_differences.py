"""Mark pixels whose RGB values differ beyond a threshold in two images.

By default, this compares the CPU SSAA-16 image with the first GPU frame.
The output starts as the first image and replaces differing pixels with pure
green.  Supply --first, --second, --output, or --threshold to reuse it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_FIRST = ROOT / "output_frames_cpu_offline_config_test" / "cpu_fullres_4096x2160_ssaa16_20260803_085157.png"
DEFAULT_SECOND = ROOT / "output_frames" / "frame_00001.png"
DEFAULT_OUTPUT = ROOT / "output_frames_cpu_offline_config_test" / "cpu_ssaa16_vs_gpu_frame00001_diff_gt2_green.png"
DEFAULT_THRESHOLD = 2


def mark_differences(first_path: Path, second_path: Path, output_path: Path, threshold: int) -> int:
    """Write a copy of the first image with substantially different pixels in green.

    A pixel is marked when at least one RGB channel differs by more than
    ``threshold``.  Alpha, when present, is retained from the first image.
    """

    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be between 0 and 255")

    first = cv2.imread(str(first_path), cv2.IMREAD_UNCHANGED)
    second = cv2.imread(str(second_path), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise FileNotFoundError(f"Unable to read first image: {first_path}")
    if second is None:
        raise FileNotFoundError(f"Unable to read second image: {second_path}")
    if first.shape != second.shape:
        raise ValueError(f"Images must have the same shape; got {first.shape} and {second.shape}")
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ValueError("Images must be BGR or BGRA colour images")

    rgb_difference = np.abs(first[..., :3].astype(np.int16) - second[..., :3].astype(np.int16))
    differing_pixels = np.max(rgb_difference, axis=2) > threshold
    result = first.copy()
    result[differing_pixels, :3] = (0, 255, 0)  # BGR pure green

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Unable to write output image: {output_path}")
    return int(np.count_nonzero(differing_pixels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=DEFAULT_FIRST, help="base image to copy")
    parser.add_argument("--second", type=Path, default=DEFAULT_SECOND, help="image to compare against")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="green-marked output PNG")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="strict RGB difference threshold")
    args = parser.parse_args()

    marked = mark_differences(args.first, args.second, args.output, args.threshold)
    print(f"Marked {marked} pixels with RGB difference > {args.threshold}; saved to {args.output}")


if __name__ == "__main__":
    main()
