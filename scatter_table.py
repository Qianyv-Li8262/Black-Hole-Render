# SPDX-License-Identifier: GPL-3.0-or-later
"""On-disk SH layout and chunked upload for the disk renderer."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from helpers.disk_cache import disk_half_extent

ROOT = Path(__file__).resolve().parent
FORMAT = "disk-scatter-incoming-sh-v1"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def solver_revision():
    h = hashlib.sha256()
    for name in ("krnls/scatter/bake.cu", "krnls/scatter/bake_ray.cuh", "krnls/scatter/scatter_sh.cuh", "krnls/disk.cu",
                 "krnls/radiative_transfer.h", "krnls/cuda_vec_math_utils.cuh"):
        h.update((ROOT / name).read_text(encoding="utf8").encode())
    return h.hexdigest()


def scene_settings(cfg, *, include_disk_hashes=True):
    if not getattr(cfg, "DISK_COEFFICIENTS", False) or (ROOT / cfg.kernel_path).resolve() != ROOT / "krnls/disk.cu":
        raise ValueError("Scatter tables require the coefficient disk.cu material")
    disk = ROOT / cfg.prebaked_disk_path
    period = float(getattr(cfg, "DISK_LOOP_PERIOD", 0.0))
    if not math.isfinite(period) or period < 0:
        raise ValueError("Invalid disk loop period")
    scene = dict(color_sha256=digest(ROOT / cfg.color_lut_path),
                 vertical_half_extent=disk_half_extent(disk), loop_period=period,
                 rgb_shift=bool(getattr(cfg, "ACCRE_DISK_DOPPLER_FOLLOW_BKGD", False)))
    if include_disk_hashes:
        scene["disk_sha256"] = digest(disk)
    if period:
        reference, epoch = float(cfg.DISK_LOOP_REFERENCE_TIME), float(cfg.DISK_LOOP_EPOCH)
        if not math.isfinite(reference) or not math.isfinite(epoch):
            raise ValueError("Disk loop reference and epoch must be finite")
        scene.update(loop_reference=reference, loop_epoch=epoch)
        if include_disk_hashes:
            scene["disk_b_sha256"] = digest(ROOT / cfg.DISK_LOOP_TEXTURE_B)
    return scene


def table_shape(grid, order):
    if type(order) is not int or not 0 <= order <= 8:
        raise ValueError("SH order must be an integer in [0, 8]")
    if len(grid) != 3 or any(type(n) is not int or n < 2 for n in grid) or math.prod(grid) >= 2**31:
        raise ValueError("Invalid (radius, height, azimuth) scatter grid")
    groups = ((order + 1)**2 + 3) // 4
    return (groups * 3, *grid, 4)


def pack_sh(coeff, planes):
    """Pack (cells, SH, RGB) coefficients into FP16 texture planes (planes, cells, 4).

    Plane 3*k+c holds SH coefficients 4k..4k+3 of RGB channel c; unused slots are zero.
    """
    cells, count, _ = coeff.shape
    packed = np.zeros((cells, planes // 3 * 4, 3), np.float16)
    packed[:, :count] = coeff
    return packed.reshape(cells, -1, 4, 3).transpose(1, 3, 0, 2).reshape(-1, cells, 4)


def read_epoch(path):
    path = Path(path)
    info = json.loads(path.read_text(encoding="utf8"))
    if info.get("format") != FORMAT or type(info.get("epoch")) is not int or info["epoch"] < 1:
        raise ValueError(f"Invalid scatter epoch in {path}")
    if not (-1 < info["g"] < 1 and 0 <= info["albedo"] <= 1):
        raise ValueError("Invalid HG phase or albedo")
    shape = table_shape(info["grid"], info["order"])
    array = np.load(path.with_name("table.npy"), mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float16 or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"Expected a contiguous FP16 scatter table of shape {shape}")
    return info, array


class ScatterTable:
    def __init__(self, info, array):
        import cupy as cp
        from cupy.cuda import runtime, texture

        self.options = ("-DDISK_SCATTER", f"-DSCATTER_ORDER={info['order']}",
                        f"-DSCATTER_SOURCE_ID=0x{solver_revision()[:8]}u")
        self.textures = []
        desc = texture.ChannelFormatDescriptor(16, 16, 16, 16, runtime.cudaChannelFormatKindFloat)
        sampler = texture.TextureDescriptor(
            addressModes=(runtime.cudaAddressModeWrap, runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
            filterMode=runtime.cudaFilterModeLinear, readMode=runtime.cudaReadModeElementType,
            normalizedCoords=1)
        if any(not np.isfinite(plane).all() for plane in array):
            raise ValueError("Scatter table contains non-finite coefficients")
        for plane in array:
            nr, nz, nphi, _ = plane.shape
            storage = texture.CUDAarray(desc, nphi, nz, nr)
            storage.copy_from(np.asarray(plane).reshape(nr, nz, nphi*4))
            self.textures.append(texture.TextureObject(texture.ResourceDescriptor(runtime.cudaResourceTypeArray,
                                                                                 cuArr=storage), sampler))
        self.handles = cp.asarray([tex.ptr for tex in self.textures], dtype=cp.uint64)
        # Finish host uploads before dropping the memory map or using another stream.
        cp.cuda.get_current_stream().synchronize()
        self.args = (self.handles, np.float32(info["g"]), np.float32(info["albedo"]))


def scatter_enabled(cfg):
    # Configs written before OPEN_SCATTER used the table path as the switch.
    return bool(getattr(cfg, "OPEN_SCATTER", getattr(cfg, "SCATTER_TABLE", None)))


def configured_table(cfg, *, enabled=None):
    if enabled is None:
        enabled = scatter_enabled(cfg)
    if not enabled:
        return None
    name = getattr(cfg, "SCATTER_TABLE", None)
    if not name:
        raise ValueError("Set SCATTER_TABLE to an epoch JSON when enabling scattering")
    info, array = read_epoch(ROOT / name)
    check_disk_hashes = getattr(cfg, "SCATTER_CHECK_DISK_HASHES", True)
    baked_scene = info["scene"]
    if not check_disk_hashes:
        baked_scene = {key: value for key, value in baked_scene.items() if key not in {"disk_sha256", "disk_b_sha256"}}
    if baked_scene != scene_settings(cfg, include_disk_hashes=check_disk_hashes):
        raise ValueError("Scatter table belongs to different disk, LUT, or loop settings")
    override = getattr(cfg, "SCATTER_G", None)
    if override is not None:
        if not -1 < override < 1:
            raise ValueError("SCATTER_G must be in (-1, 1)")
        if info["epoch"] > 1 and override != info["g"]:
            raise ValueError("Changing HG g after epoch one requires rebaking subsequent epochs")
        info = dict(info, g=float(override))
    return ScatterTable(info, array)
