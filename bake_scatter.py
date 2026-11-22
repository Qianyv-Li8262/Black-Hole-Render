# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake cumulative RGB incident radiance in a frozen disk snapshot (python bake_scatter.py)."""
import argparse
from contextlib import ExitStack, closing
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

from helpers.disk_cache import load_disk_cache, validate_disk_loop_pair
from scatter_table import FORMAT, ROOT, ScatterTable, pack_sh, read_epoch, scene_settings, solver_revision, table_shape


# Both kernels trace and store the same rays. Only the assignment to threads changed.
CELL_LOOP_REVISIONS = {
    "dcf851faf0132c9e83a33a91ce9d9a5ffb092f70b0005e0db4061838e0980a8b",
    "4019b8c9fca74a274f523a370d44fc278eb0d8834063a1825c9ae3d92e70f796",
}


def quadrature(count, order):
    # Uniform longitude integrates Fourier modes; GL nodes integrate in cos(theta).
    rings = [n for n in range(order + 1, math.isqrt(count) + 1)
             if count % n == 0 and count // n >= 2*order + 1]
    if not rings:
        raise ValueError("Directions must factor into >= L+1 rings and >= 2L+1 azimuths (e.g. 256)")
    ntheta = min(rings, key=lambda n: abs(math.log(count / (2*n*n))))
    nphi = count // ntheta
    mu, weights = np.polynomial.legendre.leggauss(ntheta)
    phi = (np.arange(nphi) + 0.5) * (2*np.pi/nphi)
    radial = np.sqrt(1 - mu*mu)[:, None]
    q = np.empty((ntheta, nphi, 4), np.float32)
    q[..., 0] = radial * np.cos(phi)
    q[..., 1] = radial * np.sin(phi)
    q[..., 2] = mu[:, None]
    q[..., 3] = weights[:, None] * (2*np.pi/nphi)
    return q.reshape(-1, 4)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", nargs=3, type=int, default=[512, 32, 512], metavar=("R", "Z", "PHI"))
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--directions", type=int, default=256)
    p.add_argument("--g", type=float, default=0.8, help="HG mean scattering cosine, -1 < g < 1")
    p.add_argument("--albedo", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=1, help="Last cumulative epoch number to write")
    p.add_argument("--keep-epochs", action="store_true", help="Keep every completed epoch instead of rolling checkpoints")
    p.add_argument("--chunk", type=int, help="Cells per launch (default up to 16384 cells and 4M rays)")
    p.add_argument("--time", type=float, help="Frozen coordinate time (default cfg.start_t)")
    p.add_argument("--step", type=float, help="Base RK step (default cfg.step)")
    p.add_argument("--maxstep", type=int, help="Ray integration budget (default cfg.maxstep)")
    p.add_argument("--output", type=Path, default=ROOT / "output_frames/scatter")
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path, help="Completed epoch JSON with matching settings")
    resume.add_argument("--resume-latest", action="store_true", help="Continue the latest epoch in --output, or start if empty")
    args = p.parse_args()
    if args.directions < 1:
        p.error("--directions must be positive")
    if args.chunk is None:
        args.chunk = min(16384, max(1, (4 << 20) // args.directions))
    if args.chunk < 1 or args.epochs < 1 or args.chunk*args.directions >= 2**31:
        p.error("Invalid directions, chunk size or epoch count")
    if not (-1 < args.g < 1 and 0 <= args.albedo <= 1):
        p.error("Require -1 < g < 1 and 0 <= albedo <= 1")
    return args


def epoch_folders(output):
    for folder in output.iterdir():
        match = re.fullmatch(r"epoch_([0-9]{4,})", folder.name)
        if match and folder.is_dir():
            yield int(match[1]), folder


def epoch_info(path):
    info, array = read_epoch(path)
    array._mmap.close()
    return info


def latest_epoch(output):
    if not output.exists():
        return None
    candidates = list(epoch_folders(output))
    if not candidates:
        return None
    number, folder = max(candidates)
    path = folder / "epoch.json"
    info = epoch_info(path)
    if info["epoch"] != number:
        raise ValueError(f"Epoch number does not match its directory: {path}")
    return path


def prune_rolling_epochs(output, newest_epoch):
    root = output.resolve()
    for number, folder in list(epoch_folders(output)):
        if number >= newest_epoch or folder.is_symlink() or folder.resolve().parent != root:
            continue
        manifest, table = folder / "epoch.json", folder / "table.npy"
        try:
            if manifest.is_symlink() or table.is_symlink() or not manifest.is_file():
                continue
            info = json.loads(manifest.read_text(encoding="utf8"))
            if (not isinstance(info, dict) or info.get("format") != FORMAT
                    or info.get("epoch") != number or info.get("checkpoint_policy") != "rolling"):
                continue
            if any(entry not in (manifest, table) for entry in folder.iterdir()):
                continue
            table.unlink(missing_ok=True)
            manifest.unlink()
            folder.rmdir()
        except (OSError, ValueError) as error:
            print(f"Could not remove old scatter checkpoint {folder}: {error}", file=sys.stderr, flush=True)


def reserve_epoch_file(path, byte_count):
    # On Linux, a sparse mmap can raise SIGBUS on write when the backing disk fills.
    free = shutil.disk_usage(path.parent).free
    if free < byte_count:
        raise OSError(errno.ENOSPC, f"scatter epoch needs {byte_count} bytes; {free} bytes free", str(path))
    if hasattr(os, "posix_fallocate"):
        with path.open("r+b") as stream:
            try:
                os.posix_fallocate(stream.fileno(), 0, os.fstat(stream.fileno()).st_size)
            except OSError as error:
                if error.errno not in (errno.ENOSYS, errno.EOPNOTSUPP):
                    raise


def publish_epoch(stage, destination):
    for attempt in range(6):
        try:
            stage.rename(destination)
            return
        except PermissionError as error:
            # A short-lived Windows file handle can block the directory rename.
            if (getattr(error, "winerror", None) not in (5, 32, 33)
                    or destination.exists() or attempt == 5):
                raise
            time.sleep(0.05 * 2**attempt)


def main():
    args = arguments()
    import cupy as cp

    with ExitStack() as dll_paths:
        try:
            cp.cuda.device.get_cublas_handle()
        except ImportError:
            if os.name != "nt":
                raise
            # Windows can point CUDA_PATH at a newer toolkit than the CuPy wheel.
            major = cp.cuda.runtime.runtimeGetVersion() // 1000
            versions = [(int(key.rsplit("_", 1)[-1]), Path(root))
                        for key, root in os.environ.items()
                        if key.startswith(f"CUDA_PATH_V{major}_") and key.rsplit("_", 1)[-1].isdigit()]
            for _, root in sorted(versions, reverse=True):
                folder = root / "bin"
                dll = f"cublas64_{major}.dll"
                if (folder / "x64" / dll).is_file():
                    folder /= "x64"
                if (folder / dll).is_file():
                    dll_paths.enter_context(os.add_dll_directory(str(folder)))
                    break
            cp.cuda.device.get_cublas_handle()
        bake(args, cp)


def bake(args, cp):
    shape = table_shape(args.grid, args.order)
    q = quadrature(args.directions, args.order)
    from cfg import offline as cfg
    from helpers.cuda_tex import create_texture_array_2d, create_texture_array_3d
    scene = scene_settings(cfg)
    instant = cfg.start_t if args.time is None else args.time
    step = cfg.step if args.step is None else args.step
    maxstep = cfg.maxstep if args.maxstep is None else args.maxstep
    if not np.isfinite([instant, step]).all() or step <= 0 or maxstep < 1:
        raise ValueError("Invalid snapshot time, step, or integration budget")
    revision = solver_revision()
    options = [f"-DSCATTER_SOURCE_ID=0x{revision[:8]}u", "-DDISK_SCATTER", "-DDISK_COEFFICIENTS", "-DNO_DEPTH_JITTER",
               f"-DSCATTER_ORDER={args.order}", f"-DDISK_HALF_EXTENT={scene['vertical_half_extent']}f",
               f"-I{ROOT / 'krnls'}", "-use_fast_math"]
    for flag in ("USE_RK4", "PHOTON_RING_OPT", "ACCRE_DISK_DOPPLER_FOLLOW_BKGD"):
        if getattr(cfg, flag, False):
            options.append("-D" + flag)
    settings = dict(format=FORMAT, solver_revision=revision, scene=scene, grid=args.grid, order=args.order, directions=args.directions,
                    g=args.g, albedo=args.albedo, time=float(instant), step=float(step), maxstep=int(maxstep),
                    rk4=bool(getattr(cfg, "USE_RK4", False)), photon_ring=bool(getattr(cfg, "PHOTON_RING_OPT", False)))
    first = 1
    previous_path = latest_epoch(args.output) if args.resume_latest else args.resume
    if previous_path:
        meta = epoch_info(previous_path)
        changeable = {"g", "albedo"} if meta["epoch"] == 1 and not args.resume_latest else set()
        if revision in CELL_LOOP_REVISIONS and meta.get("solver_revision") in CELL_LOOP_REVISIONS:
            changeable.add("solver_revision")
        if any(meta.get(k) != v for k, v in settings.items() if k not in changeable):
            raise ValueError("Resume settings differ from the supplied epoch")
        first = meta["epoch"] + 1
        if args.resume_latest and not args.keep_epochs:
            prune_rolling_epochs(args.output, meta["epoch"])
    if args.epochs < first:
        if args.resume_latest:
            print(f"Requested epoch {args.epochs} is already complete: {previous_path}", flush=True)
            return
        raise ValueError("--epochs must exceed the resumed epoch")
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(first, args.epochs + 1):
        if (args.output / f"epoch_{i:04d}").exists():
            raise FileExistsError(f"Epoch {i} already exists in {args.output}")

    def texture(path):
        a = load_disk_cache(path, coefficients=True)
        return create_texture_array_3d(cp.asarray(a), 4, (0, 0, 1, 1, 1))

    disk_path = ROOT / cfg.prebaked_disk_path
    loop_args = ()
    if scene["loop_period"]:
        b_path = ROOT / cfg.DISK_LOOP_TEXTURE_B
        validate_disk_loop_pair(disk_path, b_path)
        disk_b = texture(b_path)
        options.append("-DDISK_LOOP")
        period, reference = scene["loop_period"], scene["loop_reference"]
        qa = ((instant - scene["loop_epoch"]) / period + 0.5) % 1.0
        qb = (qa + 0.5) % 1.0
        loop_args = (np.uint64(disk_b.ptr), np.float32(reference + period*(qa-0.5)),
                     np.float32(reference + period*(qb-0.5)), np.float32(0.5 - 0.5*math.cos(2*math.pi*qa)))
    disk = texture(disk_path)
    lut = np.load(ROOT / cfg.color_lut_path)
    rgba = np.ones((*lut.shape[:-1], 4), np.float16)
    rgba[..., :3] = lut
    color = create_texture_array_2d(cp.asarray(rgba), 4, (1, 1, 1, 1), True)
    module = cp.RawModule(code=(ROOT / "krnls/scatter/bake.cu").read_text(encoding="utf8"), options=tuple(options))
    trace = module.get_function("bake_rays")
    project = module.get_function("projection_kernel")
    dirs = cp.asarray(q)
    nr, nz, nphi = args.grid
    phi = (np.arange(nphi) + 0.5) * (2*np.pi/nphi)
    azimuths = cp.asarray(np.column_stack((np.cos(phi), np.sin(phi))), dtype=cp.float32)
    count = (args.order + 1)**2
    P = cp.empty((count, len(q)), np.float32)
    project(((len(q)+127)//128,), (128,), (dirs, P, np.int32(len(q))))
    bands = np.floor(np.sqrt(np.arange(count))).astype(int)
    phase_squared = (args.g ** bands)**2
    size = nr*nz*nphi
    table_bytes = math.prod(shape) * np.dtype(np.float16).itemsize
    capacity = min(size, args.chunk)
    ray_buffer = cp.empty((capacity, len(q), 3), cp.float32)
    # Both products of a batch travel to the CPU in one contiguous copy.
    readback = cp.empty(capacity * (3*count + 1), cp.float32)
    grid_args = tuple(np.int32(n) for n in args.grid)
    for epoch in range(first, args.epochs + 1):
        started = time.perf_counter()
        old = old_array = None
        if previous_path:
            meta, old_array = read_epoch(previous_path)
            old = ScatterTable(meta, old_array)
        ptr = old.handles if old else np.uint64(0)
        truncated = cp.zeros(1, np.uint64)
        delta2 = total2 = 0.0
        folder = args.output / f"epoch_{epoch:04d}"
        # Publish the table and JSON together. A partial epoch is never a resume point.
        with tempfile.TemporaryDirectory(prefix=".scatter-", dir=args.output) as scratch:
            stage = Path(scratch) / folder.name
            stage.mkdir()
            table_path = stage / "table.npy"
            out = np.lib.format.open_memmap(table_path, mode="w+", dtype=np.float16, shape=shape)
            with closing(out._mmap):
                reserve_epoch_file(table_path, table_bytes)
                planes = out.reshape(shape[0], size, 4)
                prev = old_array.reshape(shape[0], size, 4) if old_array is not None else None
                for start in range(0, size, args.chunk):
                    cells = min(args.chunk, size-start)
                    rays = ray_buffer[:cells]
                    ncoeff = cells * count * 3
                    batch = readback[:ncoeff + cells]
                    sh = batch[:ncoeff].reshape(cells, count, 3)
                    weights = batch[ncoeff:]
                    trace(((cells+127)//128,), (128,),
                          (rays, dirs, azimuths, np.int32(start), np.int32(cells), np.int32(len(q)),
                           *grid_args, np.uint64(disk.ptr), np.uint64(color.ptr), np.float32(instant),
                           np.float32(step), np.int32(maxstep), *loop_args, ptr,
                           np.float32(args.g), np.float32(args.albedo), weights, truncated))
                    cp.matmul(P, rays, out=sh)
                    host = batch.get()
                    coeff = host[:ncoeff].reshape(cells, count, 3)
                    if not np.isfinite(coeff).all() or np.max(np.abs(coeff)) > 65504:
                        raise FloatingPointError(f"Unrepresentable SH coefficient at cell batch {start}")
                    # Compute RMS on the quantized table actually consumed by the next epoch.
                    planes[:, start:start+cells] = pack_sh(coeff, shape[0])
                    vol = host[ncoeff:].astype(np.float64)
                    if not np.isfinite(vol).all():
                        raise FloatingPointError("Invalid scattering optical weight")
                    weighted = vol[:, None, None] * phase_squared[None, :, None]
                    current = coeff.astype(np.float16).astype(np.float32)
                    total2 += np.sum(weighted * current**2)
                    if prev is not None:
                        prior = prev[:, start:start+cells].reshape(-1, 3, cells, 4)
                        prior = prior.transpose(2, 0, 3, 1).reshape(cells, -1, 3)
                        delta2 += np.sum(weighted * (current - prior[:, :count])**2)
                out.flush()
            if prev is None:
                delta2 = total2
            rms = math.sqrt(delta2/total2) if total2 else (0.0 if not delta2 else None)
            result = dict(settings, epoch=epoch, previous_epoch=epoch-1, rays=size*len(q),
                          truncated_rays=int(truncated.get()[0]), relative_scatter_rms=rms,
                          checkpoint_policy="keep" if args.keep_epochs else "rolling",
                          elapsed_seconds=time.perf_counter()-started)
            (stage / "epoch.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
            publish_epoch(stage, folder)
        previous_path = folder / "epoch.json"
        print(f"Epoch {epoch}: {previous_path}  RMS={rms}  truncated rays={result['truncated_rays']}", flush=True)
        del old, prev, ptr
        if old_array is not None:
            old_array._mmap.close()
        if not args.keep_epochs:
            prune_rolling_epochs(args.output, epoch)


if __name__ == "__main__":
    main()
