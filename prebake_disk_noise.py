"""
预烘焙吸积盘噪声纹理。

将 blackholekernel3_moving disk speed.cu 中第 348-377 行的逻辑：
  1. tex2D(lut_physics)      -> 读取原始物理参数 (density, temp, intensity)
  2. repeaterTurbulence(...)  -> 计算 3D 程序化噪声
  3. parameters *= exp(noise) -> 噪声调制

合并预计算为一个 3D 纹理，柱坐标索引 (r_disk, |z|, phi)。
运行时 kernel 用一次 tex3D 查表替代上述三步骤。
g 因子保持运行时计算。

输入 lut_physics 使用硬件纹理双线性采样（与运行时 kernel 一致）。
"""

import numpy as np
import cupy as cp
from cupy.cuda import texture, runtime
import os

# ============================================================
# 纹理分辨率
# ============================================================
R_SAMPLES   = 256   # r_disk: 4.9495 .. 25.0
Z_SAMPLES   = 128    #  z:    -2.5    .. 2.5
PHI_SAMPLES = 128   # phi:    0.0    .. 2*pi

R_DISK_MIN = 4.9495
R_DISK_MAX = 25.0
Z_MAX = 2.5

# ============================================================
# 工具：创建 2D 纹理对象（refer to schwarschild try 2_offline_render.py）
# ============================================================
def create_2d_texture(arr):
    """arr: CuPy array, shape (h, w, c), float32."""
    h, w, c = arr.shape
    bytes_per_pixel = 16  # 4 * float32
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :c] = arr

    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32,
                                             runtime.cudaChannelFormatKindFloat)
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D,
        arr=rgba,
        chDesc=ch_fmt,
        width=w,
        height=h,
        pitchInBytes=pitch_bytes,
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1,
    )
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    return tex_obj


# ============================================================
# CuPy RawKernel
# ============================================================
KERNEL_CODE = r'''
#include "cuda_noise.cuh"

extern "C" __global__
void prebake_disk_kernel(
    cudaTextureObject_t lut_physics,     // 2D 纹理: (z, r) 硬件双线性采样
    float* __restrict__ output,          // flat: (r_out * z_out * phi_out * 4)
    float r_min, float r_max, float z_max,
    int r_out, int z_out, int phi_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = r_out * z_out * phi_out;
    if (tid >= total) return;

    int phi_idx = tid % phi_out;
    int tmp     = tid / phi_out;
    int z_idx   = tmp % z_out;
    int r_idx   = tmp / z_out;

    float r_disk = r_min + (r_max - r_min) * ((float)r_idx + 0.5f) / (float)r_out;
    float z_abs  = (((float)z_idx + 0.5f) / (float)z_out-0.5f) * z_max*2.0f;
    float phi    = ((float)phi_idx + 0.5f) / (float)phi_out * 6.283185307f;

    float r_3d = sqrtf(r_disk * r_disk + z_abs * z_abs);

    float3 noise_pos = make_float3(
        r_3d * cosf(phi),
        r_3d * sinf(phi),
        z_abs * 5.0f
    );

    // 与原 CUDA kernel line 371 完全一致
    float noise = cudaNoise::repeaterTurbulence(
        noise_pos, 2.0f,2.0f, 114514, 0.25f, 4,
        cudaNoise::BASIS_SIMPLEX, cudaNoise::BASIS_SIMPLEX
    ) * 0.8f;

    // ---- 硬件双线性采样 lut_physics ----
    //   归一化纹理坐标: u = (r_disk - r_min) / (r_max - r_min)
    //                   v = z_abs / z_max
    float tex_u = (r_disk - r_min) / (r_max - r_min);
    float tex_v = fabsf(z_abs) / z_max;
    float4 params = tex2D<float4>(lut_physics, tex_u, tex_v);

    float density   = params.x * __expf(noise);
    float temp      = params.y * __expf(noise * 0.67f);
    float intensity = params.z * __expf(noise * 2.67f);

    int out_off = (r_idx * z_out * phi_out + z_idx * phi_out + phi_idx) * 4;
    output[out_off + 0] = density;
    output[out_off + 1] = temp;
    output[out_off + 2] = intensity;
    output[out_off + 3] = 0.0f;
}
'''


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载 lut_physics 并创建纹理
    print("加载 disk_lut_for_mov_disk.npy ...")
    lut = np.load(os.path.join(base_path, 'disk_lut_for_mov_disk.npy')).astype(np.float32)
    print(f"  尺寸: {lut.shape}")

    # 转置为 (h, w, c) = (z_pix, r_pix, 4) 适配 create_2d_texture
    lut_gpu = cp.asarray(lut)
    tex_lut = create_2d_texture(lut_gpu)
    print("  lut_physics 纹理对象已创建")

    # 2. 分配输出
    total_voxels = R_SAMPLES * Z_SAMPLES * PHI_SAMPLES
    output_flat = cp.empty((total_voxels * 4,), dtype=cp.float32)

    print(f"\n预烘焙 3D 噪声纹理...")
    print(f"  分辨率: {R_SAMPLES} (r) × {Z_SAMPLES} (z) × {PHI_SAMPLES} (phi)")
    print(f"  总像素: {total_voxels:,} ({total_voxels * 4 * 4 / 1024 / 1024:.1f} MB)")

    # 3. 编译并运行
    kernel = cp.RawKernel(
        KERNEL_CODE, 'prebake_disk_kernel',
        options=(f'-I{base_path}',)
    )

    block = 256
    grid = ((total_voxels + block - 1) // block,)

    kernel(
        grid, (block,),
        (
            np.uint64(tex_lut.ptr),
            output_flat,
            np.float32(R_DISK_MIN), np.float32(R_DISK_MAX), np.float32(Z_MAX),
            np.int32(R_SAMPLES), np.int32(Z_SAMPLES), np.int32(PHI_SAMPLES),
        )
    )
    cp.cuda.Device().synchronize()

    # 4. 保存
    result = output_flat.get().reshape(R_SAMPLES, Z_SAMPLES, PHI_SAMPLES, 4)
    out_path = os.path.join(base_path, 'prebaked_disk_noise.npy')
    np.save(out_path, result)

    print(f"\n已保存到 prebaked_disk_noise.npy")
    print(f"  Shape:  {result.shape}")
    print(f"  dtype:  {result.dtype}")
    print(f"  density   范围: [{result[..., 0].min():.4f}, {result[..., 0].max():.4f}]")
    print(f"  temp      范围: [{result[..., 1].min():.1f}, {result[..., 1].max():.1f}]")
    print(f"  intensity 范围: [{result[..., 2].min():.4f}, {result[..., 2].max():.4f}]")

    # 抽样检查
    mid_z, mid_phi = Z_SAMPLES // 2, PHI_SAMPLES // 4
    sl = result[:, mid_z, mid_phi, :]
    print(f"\n  抽样 [r, z_id={mid_z}, phi_id={mid_phi}]:")
    print(f"    r_min: dens={sl[0, 0]:.4f}  temp={sl[0, 1]:.1f}  inten={sl[0, 2]:.4f}")
    print(f"    r_mid: dens={sl[R_SAMPLES // 2, 0]:.4f}  temp={sl[R_SAMPLES // 2, 1]:.1f}  inten={sl[R_SAMPLES // 2, 2]:.4f}")
    print(f"    r_max: dens={sl[-1, 0]:.4f}  temp={sl[-1, 1]:.1f}  inten={sl[-1, 2]:.4f}")


if __name__ == '__main__':
    main()
