import cupy as cp
import numpy as np
import os
# ============================================================
# 填写你的 CUDA 源文件路径。
# 该文件中应包含：
#   extern "C" __global__ void get_cloud_density(...)
#   extern "C" __global__ void compute_density_gradient(...)
# ============================================================
CUDA_FILE = r"./prebake_cloud_params.cu"  # 例如：r"./cloud_density.cu"

# LUT 空间尺寸，数组最终 shape 为：
# (phi_samples, z_samples, r_samples, 4)
phi_samples = 256
z_samples = 128
r_samples = 256

# 物理范围
r_min = 4.95
r_max = 50.0
z_box_max = 5.0

seed = 114514

# CUDA block / grid 配置。
# 三维线程分别对应 phi, z, r。
block = (8, 8, 4)
grid = (
    (phi_samples + block[0] - 1) // block[0],
    (z_samples + block[1] - 1) // block[1],
    (r_samples + block[2] - 1) // block[2],
)
base_path = os.path.dirname(os.path.abspath(__file__))
# 读取 CUDA 源码。
with open(CUDA_FILE, "r", encoding="utf-8") as f:
    cuda_source = f.read()

# 编译 CUDA RawModule。
# 若 cuda_noise.cuh 与 CUDA_FILE 位于同一目录，通常可以正常找到。
# 若不在同一目录，需要通过 options 增加 include 路径，例如：
# options=("-I/path/to/cuda/includes",)
module = cp.RawModule(
    code=cuda_source,
    options=("--std=c++14",f'-I{base_path}'),
    name_expressions=(
        "get_cloud_density",
        "compute_density_gradient",
    ),
)

get_cloud_density = module.get_function("get_cloud_density")
compute_density_gradient = module.get_function("compute_density_gradient")

# shape: (phi, z, r, channel)
# channel 0: 温度
# channel 1: 密度
# channel 2: 光强
# channel 3: 密度梯度绝对值和
output = cp.empty(
    (phi_samples, z_samples, r_samples, 4),
    dtype=cp.float32,
)

# 第一阶段：生成温度、密度、光强。
get_cloud_density(
    grid,
    block,
    (
        output,
        np.float32(r_min),
        np.float32(r_max),
        np.float32(z_box_max),
        np.int32(phi_samples),
        np.int32(z_samples),
        np.int32(r_samples),
        np.int32(seed),
    ),
)

# 第二阶段：从 density channel（第 1 通道）计算梯度，
# 并写入 channel 3。
compute_density_gradient(
    grid,
    block,
    (
        output,
        np.float32(r_min),
        np.float32(r_max),
        np.float32(z_box_max),
        np.int32(phi_samples),
        np.int32(z_samples),
        np.int32(r_samples),
    ),
)

# 确保 GPU kernel 已经执行完毕。
cp.cuda.Stream.null.synchronize()

# 转回 NumPy 并保存为 .npy。
output_numpy = cp.asnumpy(output.transpose(2, 1, 0, 3).astype(cp.float16))

np.save("accretion_disk_lut.npy", output_numpy)

print("Saved:", "accretion_disk_lut.npy")
print("Shape:", output_numpy.shape)
print("Dtype:", output_numpy.dtype)