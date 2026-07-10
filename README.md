<img width="3200" height="2000" alt="frame_00235" src="https://github.com/user-attachments/assets/d4b0b310-6301-46d4-bbc1-f1a893aeedad" />
<img width="3200" height="2000" alt="frame_00001" src="https://github.com/user-attachments/assets/4887c771-e268-489d-a3fc-232ec887c911" />


## A simple CUDA blackhole renderer

A simple CUDA blackhole renderer, very lightweight with a not-bad result.

## 环境

### 离线渲染(最小配置)

不需要 CUDA Toolkit,只装 GPU 驱动即可。

```powershell
conda create -n blackhole python=3.10
conda activate blackhole
pip install cupy-cuda12x numpy opencv-python
```

### 实时预览(需要 CUDA-GL 互操作)

额外需要 `cudart64_*.dll`,装 CUDA Toolkit 12.x 即可(实际只用到了其中的 CUDA Runtime,其他组件如 cuBLAS / nvcc / Nsight 本项目均不涉及)。

```powershell
conda activate blackhole
pip install glfw PyOpenGL
```

### 硬件要求

- NVIDIA GPU(Compute Capability ≥ 5.0)
- 驱动 ≥ 545

### 运行

```powershell
# 离线渲染(输出到 output_frames/)
python "main_new_bloom copy.py"

# 实时预览
python "main_realtime copy.py"
```

参数通过 `config_offline.py` 调整。
