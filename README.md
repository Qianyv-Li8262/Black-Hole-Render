![Current render with background Doppler](docs/images/frame_00001.png)
![](docs/images/frame_00002.png)
![](docs/images/frame_00003.png)
## A simple CUDA black-hole renderer

A lightweight CUDA renderer with single-GPU, hybrid GPU/CPU, and realtime
render paths. The scripts stay independent and can be read or modified directly.

## Setup

### Sky background

The default renderer uses NASA SVS's **An Elsewhere Starfield**:

- Source: https://svs.gsfc.nasa.gov/4856/
- File: `starmap_random_2020_16k.exr`

Download the 16K EXR and place it at:

```text
assets/starmap_random_2020_16k.exr
```

Please keep the NASA SVS attribution when redistributing screenshots or derived
material that uses this background.

### License

The repository's source code is licensed under the GNU General Public License,
version 3 or later (`GPL-3.0-or-later`); see [LICENSE](LICENSE). This does not
apply to the NASA SVS assets in `assets/`, which retain their original terms and
required attribution.

### Offline rendering

The normal offline path does not require a system-wide CUDA Toolkit installation.
A recent NVIDIA driver plus the CUDA-enabled CuPy wheel with CUDA Toolkit
components is enough.

```powershell
conda create -n blackhole python=3.10
conda activate blackhole
pip install "cupy-cuda12x[ctk]" numpy opencv-python OpenEXR
```

Copy `cfg/offline_example.py` to the ignored `cfg/offline.py` and edit the
values there before rendering.

### Realtime preview on Windows

CUDA/OpenGL interop also needs `cudart64_*.dll`. Installing CUDA Toolkit 12.x
provides the runtime used by the preview.

```powershell
conda activate blackhole
pip install glfw PyOpenGL
```

### CPU accelerator on Windows or Linux HPC

The hybrid renderer can assign frames to one CPU worker alongside the GPUs.
Build the native module first:

```bash
pip install pybind11
python tools/build_cpu_render.py
```

`clang++` and `lld` are required by the current build script.
The generated `cpu_render_native*.pyd` or `.so` is stored under `build/`.

The production CPU renderer requires **AVX2**. If **AVX-512F** and
**AVX-512DQ** are available, the AVX-512 backend is selected automatically.

## Hardware

- NVIDIA GPU with Compute Capability 5.0 or newer
- NVIDIA driver 545 or newer
- CPU accelerator: x86-64 CPU with AVX2
- Optional CPU fast path: AVX-512F + AVX-512DQ

## Commands

```bash
# Simple single-GPU offline renderer
python render_single.py

# Frame-parallel renderer using all GPUs
python render_hybrid.py --gpus all

# Add a CPU accelerator to the GPU workers
python render_hybrid.py --gpus all --use-cpu --cpu-cores 112

# Full-resolution CPU test using cfg/offline.py
python render_cpu.py

# Realtime CUDA/OpenGL preview
python render_realtime.py
```

Offline runs write a timestamped `run_*.json` beside their frames so camera,
sampling, input paths, devices, and timing are not lost.

CUDA sources are under `krnls/`, sky/background textures are under `assets/`,
and LUT or prebaked NumPy arrays are under `cache/`. Rendered frames, profiling
reports, `archive/`, and `failed/` remain local data ignored by Git. Paths in
`cfg/offline.py` are relative to the repository root.

## Experimental paths

The repository also contains experimental cloud-volume rendering code and
prebaking tools. These are kept for development and reference, but are **not a
supported entry point** and are not expected to work from a fresh clone without
additional local assets or setup.

## Troubleshooting

If OpenCV reports that the OpenEXR codec is disabled, enable it before launching
the renderer.

PowerShell:

```powershell
$env:OPENCV_IO_ENABLE_OPENEXR="1"
python render_single.py
```

Linux/macOS shell:

```bash
OPENCV_IO_ENABLE_OPENEXR=1 python render_single.py
```

## 中文快速开始

默认背景使用 NASA SVS 的 **An Elsewhere Starfield**。下载
`starmap_random_2020_16k.exr` 后放到：

```text
assets/starmap_random_2020_16k.exr
```

来源页面：

https://svs.gsfc.nasa.gov/4856/

离线渲染只需要兼容的 NVIDIA 驱动和带 CUDA Toolkit 组件的 CuPy wheel，
通常不需要在系统中单独安装完整 CUDA Toolkit：

```powershell
conda create -n blackhole python=3.10
conda activate blackhole
pip install "cupy-cuda12x[ctk]" numpy opencv-python OpenEXR
```

随后将 `cfg/offline_example.py` 复制为 `cfg/offline.py` 并修改参数：

```powershell
python render_single.py
python render_hybrid.py --gpus all
python render_hybrid.py --gpus all --use-cpu --cpu-cores 112
python render_cpu.py
python render_realtime.py
```

CPU 原生后端最低要求 AVX2；若处理器支持 AVX-512F 与 AVX-512DQ，会自动
选择 AVX-512 路径。实时预览额外需要 CUDA/OpenGL interop、`glfw` 和
`PyOpenGL`。

仓库中的 cloud-volume 路径属于实验代码，不作为支持的 fresh-clone 入口。
