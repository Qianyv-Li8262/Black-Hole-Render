<img width="3200" height="2000" alt="frame_00235" src="https://github.com/user-attachments/assets/d4b0b310-6301-46d4-bbc1-f1a893aeedad" />
<img width="3200" height="2000" alt="frame_00001" src="https://github.com/user-attachments/assets/4887c771-e268-489d-a3fc-232ec887c911" />


## A simple CUDA black-hole renderer

A lightweight CUDA renderer with single-GPU, hybrid GPU/CPU, realtime, and
cloud-volume render paths. The scripts stay independent and can be read or
modified directly.

## Setup

### Offline rendering

The CUDA Toolkit is not required for the normal offline path. A recent NVIDIA
driver and the CUDA-enabled CuPy wheel are enough.

```powershell
conda create -n blackhole python=3.10
conda activate blackhole
pip install cupy-cuda12x numpy opencv-python OpenEXR
```

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

## Hardware

- NVIDIA GPU with Compute Capability 5.0 or newer
- NVIDIA driver 545 or newer

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

# Cloud-volume experiment
python render_cloud.py
```

Copy `cfg/offline_example.py` to the ignored `cfg/offline.py` and edit
the values there. Offline and cloud runs write a timestamped `run_*.json`
beside their frames so camera, sampling, input paths, devices, and timing are
not lost.

CUDA sources are under `krnls/`, sky/background textures are under `assets/`,
and LUT or prebaked NumPy arrays are under `cache/`. Rendered frames, profiling
reports, `archive/`, and `failed/` remain local data ignored by Git. Paths in
`cfg/offline.py` are relative to the repository root.

## 中文快速开始

离线渲染只需要 NVIDIA 驱动和 `cupy-cuda12x`，通常不需要安装完整 CUDA
Toolkit。复制 `cfg/offline_example.py` 为 `cfg/offline.py` 后修改参数：

```powershell
python render_single.py
python render_hybrid.py --gpus all
python render_hybrid.py --gpus all --use-cpu --cpu-cores 112
python render_cpu.py
python render_realtime.py
python render_cloud.py
```

实时预览需要 CUDA Toolkit 12.x 提供的 CUDA Runtime，以及 `glfw` 和
`PyOpenGL`。每次离线或云渲染会在输出目录保存 `run_*.json` 参数记录。
