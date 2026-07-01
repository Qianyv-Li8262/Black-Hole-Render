"""
预烘焙吸积盘噪声纹理 (NPGS 风格噪声, seamless 版本).

与 prebake_disk_noise_npgs.py 的区别:
  - 角向采样从"沿 noise_pos.z 走直线"改为"在 noise_pos (x,z) 平面绕圆".
    phi 走 2π 圆走一圈, noise_pos 自动闭合, 消除 2π↔0 接缝.
  - 径向/垂直坐标不变, 角向特征尺度 (d(noise_pos)/d(phi) 模长) 完全保留.
  - 副作用: x 分量含 R*cos(phi), 径向 filaments 有 ~8-12% 角向摇摆.
  - NPGS 噪声本身、exp(noise) 物理映射、所有参数、OUTPUT_CLIP 全部不变.

输出文件: prebaked_disk_noise_npgs_seamless.npy
(原 prebaked_disk_noise_npgs.npy 保持不变, 运行时需手动切换加载路径)

注意: 完全消除接缝还需要 runtime 在 3D 噪声纹理的 phi 轴使用
      cudaAddressModeWrap (而非 Clamp). Baker 这边只能让切片 0 ≈ 切片 127,
      不能让它们完全相等. 详见接缝成因分析.
"""

import numpy as np
import cupy as cp
from cupy.cuda import texture, runtime
import os

# ============================================================
# 纹理分辨率
# ============================================================
R_SAMPLES   = 256   # r_disk: 4.9495 .. 25.0
Z_SAMPLES   = 128   #  z:    -2.5    .. 2.5
PHI_SAMPLES = 128   # phi:    0.0    .. 2*pi

R_DISK_MIN = 4.9495
R_DISK_MAX = 25.0
Z_MAX = 2.5

# ============================================================
# NPGS 噪声参数 (与 prebake_disk_noise_npgs.py 完全一致, 不动)
# ============================================================
NPGS_OCTAVE_START = 2.0
NPGS_OCTAVE_END   = 11.0
NPGS_CONTRAST = 80.0
NPGS_CENTER_REFERENCE = 3.5
NPGS_CENTER_TARGET = 0.1
NPGS_AMPLITUDE = 0.2
NPGS_NOISE_CLIP = 4.0
OUTPUT_CLIP = 65000.0


# ============================================================
# 工具: 创建 2D 纹理对象
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
    int r_out, int z_out, int phi_out,
    float npgs_octave_start, float npgs_octave_end,
    float npgs_contrast,
    float npgs_center_reference, float npgs_center_target, float npgs_amplitude,
    float npgs_noise_clip,
    float output_clip
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = r_out * z_out * phi_out;
    if (tid >= total) return;

    int phi_idx = tid % phi_out;
    int tmp     = tid / phi_out;
    int z_idx   = tmp % z_out;
    int r_idx   = tmp / z_out;

    float r_disk = r_min + (r_max - r_min) * ((float)r_idx + 0.5f) / (float)r_out;
    float z_abs  = (((float)z_idx + 0.5f) / (float)z_out - 0.5f) * z_max * 2.0f;
    float phi    = ((float)phi_idx + 0.5f) / (float)phi_out * 6.283185307f;

    // ---- Seamless 角向采样 ----
    // 原方案: noise_pos.z = 0.02 * pow(r,0.7) * phi
    //   phi=0 时 z=0; phi=2π 时 z=0.04π*pow(r,0.7) (非整数, 随 r 变)
    //   -> 2pi <-> 0 处噪声值突变, 产生接缝 (npgsValueNoise3D 在 z 轴非周期)
    //
    // 新方案: 在噪声场 (x,z) 平面画半径 R=0.02*pow(r,0.7) 的圆
    //   phi 走 2π 圆走一圈, noise_pos 自动回到起点, 无缝
    //
    // 角向特征尺度保留证明:
    //   原方案 d(noise_pos.z)/d(phi) = 0.02*pow(r,0.7) = R
    //   新方案 |d(noise_pos)/d(phi)| = |(-R*sin(phi), 0, R*cos(phi))| = R
    //   完全一致 -> filament 沿轨道方向的密度/间距不变
    //
    // 副作用: x 分量含 R*cos(phi), 固定 r 转一圈 x 摆动 ±R.
    //   R / (0.1*r) = 0.2*pow(r,-0.3) ≈ 11.7% (r=5) → 8.0% (r=25)
    //   径向 filaments 有 8-12% 角向摇摆, 不再绝对径向.
    float R_circle = 0.02f * powf(r_disk, 0.7f);
    float3 noise_pos = make_float3(
        R_circle * cosf(phi),     // 径向基 + 角向圆周分量
        0.1f * z_abs,                             // 垂直: 大尺度 (不变)
        R_circle * sinf(phi)                      // 角向: 圆周分量 (闭合)
    );

    // NPGS 中心化噪声: 输出中心在 npgs_center_target, exp(noise) 中心在 1.
    // 物理映射绝热锁死, 参数与原版完全一致.
    float noise = cudaNoise::npgsAccretionDiskNoiseCentered(
        noise_pos,
        npgs_octave_start, npgs_octave_end, npgs_contrast,
        npgs_center_reference, npgs_center_target, npgs_amplitude,
        npgs_noise_clip
    );

    // ---- 硬件双线性采样 lut_physics ----
    float tex_u = (r_disk - r_min) / (r_max - r_min);
    float tex_v = fabsf(z_abs) / z_max;
    float4 params = tex2D<float4>(lut_physics, tex_u, tex_v);

    // 物理映射完全保留 (绝热锁死, 不动)
    float density   = params.x * __expf(noise);
    float temp      = params.y * __expf(noise * 0.67f);
    float intensity = params.z * __expf(noise * 2.67f);

    // 末端安全 clip: 直接限制三个输出值, 保证写入 half 纹理不溢出.
    density   = fmaxf(0.0f, fminf(density,   output_clip));
    temp      = fmaxf(0.0f, fminf(temp,      output_clip));
    intensity = fmaxf(0.0f, fminf(intensity, output_clip));

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

    lut_gpu = cp.asarray(lut)
    tex_lut = create_2d_texture(lut_gpu)
    print("  lut_physics 纹理对象已创建")

    # 2. 分配输出
    total_voxels = R_SAMPLES * Z_SAMPLES * PHI_SAMPLES
    output_flat = cp.empty((total_voxels * 4,), dtype=cp.float32)

    print(f"\n预烘焙 3D 噪声纹理 (NPGS 风格, seamless 版本)...")
    print(f"  分辨率: {R_SAMPLES} (r) x {Z_SAMPLES} (z) x {PHI_SAMPLES} (phi)")
    print(f"  总像素: {total_voxels:,} ({total_voxels * 4 * 4 / 1024 / 1024:.1f} MB)")
    print(f"  NPGS 参数 (与原版一致):")
    print(f"    octave [{NPGS_OCTAVE_START}, {NPGS_OCTAVE_END}], contrast={NPGS_CONTRAST}")
    print(f"    center_ref={NPGS_CENTER_REFERENCE}, center_target={NPGS_CENTER_TARGET}, amplitude={NPGS_AMPLITUDE}")
    print(f"    noise_clip={NPGS_NOISE_CLIP} (exp(clip*2.67)={np.exp(NPGS_NOISE_CLIP*2.67):.2e})")
    print(f"    output_clip={OUTPUT_CLIP} (half-safe, 末端硬限幅)")
    print(f"  角向采样: closed-circle in noise (x,z) plane (seamless)")

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
            np.float32(NPGS_OCTAVE_START), np.float32(NPGS_OCTAVE_END),
            np.float32(NPGS_CONTRAST),
            np.float32(NPGS_CENTER_REFERENCE), np.float32(NPGS_CENTER_TARGET),
            np.float32(NPGS_AMPLITUDE),
            np.float32(NPGS_NOISE_CLIP),
            np.float32(OUTPUT_CLIP),
        )
    )
    cp.cuda.Device().synchronize()

    # 4. 保存
    result = output_flat.get().reshape(R_SAMPLES, Z_SAMPLES, PHI_SAMPLES, 4)
    out_path = os.path.join(base_path, 'prebaked_disk_noise_npgs_seamless.npy')
    np.save(out_path, result)

    print(f"\n已保存到 prebaked_disk_noise_npgs_seamless.npy")
    print(f"  Shape:  {result.shape}")
    print(f"  dtype:  {result.dtype}")
    print(f"  density   范围: [{result[..., 0].min():.4f}, {result[..., 0].max():.4f}]")
    print(f"  temp      范围: [{result[..., 1].min():.1f}, {result[..., 1].max():.1f}]")
    print(f"  intensity 范围: [{result[..., 2].min():.4f}, {result[..., 2].max():.4f}]")

    # ---- 接缝诊断: 比较切片 0 和切片 PHI-1 在相同 (r,z) 处的差异 ----
    # seamless 修复后, 两切片的 noise_pos 仅差 2R*sin(pi/PHI), 应远小于原版.
    slice0 = result[:, :, 0, :]
    slice_last = result[:, :, PHI_SAMPLES - 1, :]
    diff_intensity = np.abs(slice0[..., 2] - slice_last[..., 2])
    print(f"\n  接缝诊断 (slice 0 vs slice {PHI_SAMPLES-1}):")
    print(f"    intensity 平均差异: {diff_intensity.mean():.4f}")
    print(f"    intensity 最大差异: {diff_intensity.max():.4f}")
    print(f"    intensity 整体 std: {result[..., 2].std():.4f}")
    print(f"    (差异/std 比值越小, 接缝越不明显; 原版此值通常 > 1)")

    # 抽样检查
    mid_z, mid_phi = Z_SAMPLES // 2, PHI_SAMPLES // 4
    sl = result[:, mid_z, mid_phi, :]
    print(f"\n  抽样 [r, z_id={mid_z}, phi_id={mid_phi}]:")
    print(f"    r_min: dens={sl[0, 0]:.4f}  temp={sl[0, 1]:.1f}  inten={sl[0, 2]:.4f}")
    print(f"    r_mid: dens={sl[R_SAMPLES // 2, 0]:.4f}  temp={sl[R_SAMPLES // 2, 1]:.1f}  inten={sl[R_SAMPLES // 2, 2]:.4f}")
    print(f"    r_max: dens={sl[-1, 0]:.4f}  temp={sl[-1, 1]:.1f}  inten={sl[-1, 2]:.4f}")


if __name__ == '__main__':
    main()
