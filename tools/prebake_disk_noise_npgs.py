# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""
预烘焙吸积盘噪声纹理 (NPGS 风格噪声版本)。

与 prebake_disk_noise.py 的区别:
  - 噪声源从 repeaterTurbulence(...) * 0.8 改为
    npgsAccretionDiskNoiseCentered(...) —— 乘性多倍频 + log 压缩 + 中心化
  - 坐标缩放改为接近 NPGS 的设计: r/z 方向大尺度, 角向小尺度
  - 厚云团预设使用三维胞状团块和限频 fBm，第四通道存致密核心比例
  - noise 中心在 0 (即 exp(noise) 中心在 1), 幅度可调, 不改变原有物理趋势

输出文件: cache/prebaked_disk_clouds.npy
关闭 CLOUD_STRUCTURE 后输出原版 cache/prebaked_disk_noise_npgs.npy。
厚云团预设配合 cfg/offline.py 中的 OPACITY_CHANGE=True 使用。
"""

import numpy as np
import cupy as cp
from cupy.cuda import texture, runtime
import os

# ============================================================
# 纹理分辨率
# ============================================================
R_SAMPLES   = 256   # r_disk: 4.9495 .. 35.0
Z_SAMPLES   = 128   #  z:    -2.5    .. 2.5
PHI_SAMPLES = 512   # 角向需要解析云团毛边，避免低分辨率烘焙变成毡子

R_DISK_MIN = 4.9495
R_DISK_MAX = 35.0
Z_MAX = 2.5

# 厚度是密度高斯包络的标准差，单位与渲染坐标一致。
CLOUD_STRUCTURE = True
CLOUD_HEIGHT_INNER = 0.22
CLOUD_HEIGHT_OUTER = 1.05
CLOUD_HEIGHT_VARIATION = 0.55  # 大尺度起伏对局部厚度的影响
CLOUD_DETAIL_STRENGTH = 0.45  # 保留少量 NPGS 细节，大尺度云团负责主体形状
CLOUD_EDGE_ROUGHNESS = 0.75  # 三维细节侵蚀云团表面，形成柔软毛边
CLOUD_EDGE_SOFTNESS = 0.22  # 密度从团块向雾气延伸的过渡宽度
CLOUD_LINK_DENSITY = 0.025  # 少量连接云丝；过高会沿视线叠成不透明的毡子
CLOUD_POROSITY = 0.90      # 三维疏密通道强度；0 保留连续云体，1 打开最深的缝隙
CLOUD_MIDPLANE_WARP = 0.35   # 云层中面起伏，单位为局部半厚度
CLOUD_OUTER_COVERAGE = 0.70     # 越小外盘越破碎；内盘保留连续亮带
CLOUD_EMISSION_GAIN = 4.0        # 在烘焙中提亮云体，保持原径向色温和背景曝光
COLD_SOURCE_BOOST = 1.0          # 让致密云面在高消光时仍能发亮
COLD_DENSITY_BOOST = 4.0      # 暗核密度最高乘 (1 + boost)
COLD_EMISSION_SUPPRESSION = 0.8  # 保留云体发光，避免高消光区变成死黑剪影
COLD_CORE_START = 0.48
COLD_CORE_END = 0.70

# ============================================================
# NPGS 噪声参数 (可在不破坏物理趋势的前提下调整)
# ============================================================
# 倍频范围: start..end. NPGS 典型 ~2..4. 越大越细, 越小越粗.
NPGS_OCTAVE_START = 2.0
NPGS_OCTAVE_END   = 11.0
# log 压缩指数: 越大云团越尖锐 (NPGS 典型 80). 保留 NPGS 原值.
NPGS_CONTRAST = 80.0
# 归一化参考: raw_n 的典型上界. contrast=80, 2倍频时 raw_n 大部分在 [0, 4],
# 长尾到 ~15. 取 4.0 把主流云团压到 norm~[0,1], 尖峰 norm>1 但不爆炸.
#   调小 -> norm 偏大 -> exp(noise*2.67) 易溢出 inf (曾导致黑屏)
#   调大 -> norm 偏小 -> 扰动幅度不足
NPGS_CENTER_REFERENCE = 3.5
# DC 偏移: 必须近似 norm=raw_n/center_reference 的*均值*, 不是 0!
# raw_n 是右偏长尾分布, norm 均值约 0.35. 减去它才真正零均值.
#   设 0 -> 输出残留正 DC -> exp(noise*2.67) 溢出 inf -> 黑屏 (之前的 bug)
NPGS_CENTER_TARGET = 0.1
# 输出幅度: 与原 repeaterTurbulence*0.8 的 ~[-0.8, 0.8] 量级可比.
#   调大 -> 扰动更剧烈; 调小 -> 更柔和.
NPGS_AMPLITUDE = 0.2
# 安全硬上限: 对称 clamp(noise, -clip, +clip). 防止 exp(noise*2.67) 溢出 inf.
#   exp(clip*2.67) 上限: clip=4 -> 4.4e4 (HDR 安全); clip=10 -> 2.6e11 (仍 float 安全)
#   clip 上限受 float32 限制: noise < 33 才不溢出. 取 4.0 给 bloom 足够 headroom.
#   自由调参时这个值兜底, amplitude 再大、center_reference 再小也不会 inf.
#   注意: noise_clip 只限扰动本身, 最终输出值由 OUTPUT_CLIP 兜底 (见 kernel 末端).
NPGS_NOISE_CLIP = 4.0

# 噪声种子: 改这个值就换一张盘 (任意整数).
# 透传到 npgsAccretionDiskNoiseCentered -> ... -> randomGrid(x,y,z,seed),
# seed 作为整数偏移注入位置哈希, 换值即得不同的 lattice 格点布局.
# 0 = 原始盘 (seed 未参与哈希时的等价态); 换任意非零整数即得新盘.
NPGS_SEED = 1919810
# 最终输出值硬上限: 对 density/temp/intensity 三个烘焙输出做 half-safe clip.
# 运行时纹理以 half 存储, 上限 ~65504. 留余量取 65000.
#   这一道兜底覆盖 noise_clip 管不到的情况 —— 当 params.z (LUT 基线) 本身很大时,
#   即使 noise 被钳在 [-4,4], params.z * exp(4*2.67) = params.z * 4.4e4 仍可能爆.
#   末端直接 clip 三个输出, 保证写入 half 纹理绝不溢出.
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

__device__ float cloud_fbm(float3 p, int seed) {
    // Only frequencies resolved by the LUT; the legacy 3^11 octave is not
    // appropriate for a texture whose azimuth has a finite sample count.
    return 0.57f * cudaNoise::simplexNoise(p, 1.0f, seed) +
           0.29f * cudaNoise::simplexNoise(p, 2.0f, seed + 101) +
           0.14f * cudaNoise::simplexNoise(p, 4.0f, seed + 203);
}

__device__ float cloud_lobe_distance(float3 p, int seed) {
    // Rounded 3D lobes. floorf is essential for continuity in negative cells.
    int ix = (int)floorf(p.x), iy = (int)floorf(p.y), iz = (int)floorf(p.z);
    float distance = 10.0f;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                unsigned int h = cudaNoise::hash((unsigned int)(ix + x) * 73856093u ^
                                                (unsigned int)(iy + y) * 19349663u ^
                                                (unsigned int)(iz + z) * 83492791u ^ (unsigned int)seed);
                float dx = p.x - (ix + x + 0.15f + 0.7f * cudaNoise::randomFloat(h));
                float dy = p.y - (iy + y + 0.15f + 0.7f * cudaNoise::randomFloat(h ^ 0x68bc21ebu));
                float dz = p.z - (iz + z + 0.15f + 0.7f * cudaNoise::randomFloat(h ^ 0x02e5be93u));
                float radius = 0.5f + 0.5f * cudaNoise::randomFloat(h ^ 0x967a889bu);
                distance = fminf(distance, sqrtf(dx*dx + dy*dy + dz*dz) / radius);
            }
    return distance;
}

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
    float output_clip,
    int npgs_seed,
    float height_inner, float height_outer, float height_variation,
    float edge_roughness, float edge_softness, float link_density, float porosity,
    float midplane_warp, float outer_coverage,
    float emission_gain, float cold_source_boost,
    float cold_density_boost, float cold_emission_suppression,
    float cold_core_start, float cold_core_end
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

    // NPGS 风格坐标: 径向和垂直大尺度, 角向小尺度 (产生沿轨道方向的丝状结构)
    // 参考 BlackHole_common.glsl DiskColor 中的采样坐标:
    //   vec3(0.1*RotPosR, 0.1*PosY, 0.02*pow(OuterR,0.7)*PosTheta)
    // 这里 OuterR 用 r_disk 近似, PosTheta 用 phi 近似.
    float3 noise_pos = make_float3(
        0.1f * r_disk,                              // 径向: 大尺度
        0.1f * z_abs,                               // 垂直: 大尺度 (原 *5.0 改为 *0.1, 消除细碎条带)
        0.02f * powf(r_disk, 0.7f) * phi            // 角向: 小尺度丝状
    );
#ifndef DISK_CLOUD_STRUCTURE
    // NPGS 中心化噪声: 输出中心在 npgs_center_target (默认 0),
    // exp(noise) 中心在 1, 不改变原有物理趋势. 幅度由 npgs_amplitude 控制.
    // noise_clip 为安全硬上限, 防止 exp(noise*2.67) 溢出 inf.
    float noise = cudaNoise::npgsAccretionDiskNoiseCentered(
        noise_pos,
        npgs_octave_start, npgs_octave_end, npgs_contrast,
        npgs_center_reference, npgs_center_target, npgs_amplitude,
        npgs_noise_clip,
        npgs_seed
    )*__saturatef(6.9f-r_disk*0.2f);
#endif

    // ---- 硬件双线性采样 lut_physics ----
    float tex_u = (r_disk - r_min) / (r_max - r_min);
    float tex_v = fabsf(z_abs) / z_max;
#ifdef DISK_CLOUD_STRUCTURE
    // Cartesian noise is periodic across phi=0/2pi. The same broad field
    // shapes the cloud height and the cold cores within that cloud.
    float flow_angle = phi + 0.7f * logf(r_disk / r_min);
    float3 cloud_pos = make_float3(0.28f * r_disk * cosf(flow_angle),
                                  0.28f * r_disk * sinf(flow_angle), 0.0f);
    float height_noise = cudaNoise::simplexNoise(cloud_pos, 1.0f, npgs_seed + 17);
    float radial_fraction = (r_disk - r_min) / (r_max - r_min);
    float height = (height_inner + (height_outer - height_inner) * powf(radial_fraction, 0.75f)) *
                   (1.0f + height_variation * height_noise);
    height = fmaxf(height, 0.01f);
    float midplane = midplane_warp * height *
                     cudaNoise::simplexNoise(cloud_pos, 1.3f, npgs_seed + 23);
    cloud_pos.z = (z_abs - midplane) * 0.65f;
    // Warp the volume in all three axes so neighboring lobes do not form a
    // regular row of spheres. Coordinates remain periodic at the phi wrap.
    float3 unwarped = cloud_pos;
    cloud_pos.x += 0.35f * cloud_fbm(unwarped, npgs_seed + 107);
    cloud_pos.y += 0.35f * cloud_fbm(unwarped, npgs_seed + 211);
    cloud_pos.z += 0.35f * cloud_fbm(unwarped, npgs_seed + 313);
    float edge_noise = cloud_fbm(cloud_pos, npgs_seed + 59);
    float q = (z_abs - midplane) / height;
    float outward = radial_fraction * radial_fraction * (3.0f - 2.0f * radial_fraction);
    float coverage = 1.05f + (outer_coverage - 1.05f) * outward;
    float body = coverage - cloud_lobe_distance(cloud_pos, npgs_seed + 31) +
                 edge_roughness * edge_noise - (0.25f - 0.12f * outward) * q * q;
    float core_noise = __saturatef(body + 0.25f);
    float cold_fraction = __saturatef((core_noise - cold_core_start) / (cold_core_end - cold_core_start));
    cold_fraction = cold_fraction * cold_fraction * (3.0f - 2.0f * cold_fraction);
    float noise = npgs_amplitude * cloud_fbm(cloud_pos, npgs_seed + 83);
    // Sample the original radial profile at the midplane, then apply the
    // new height envelope instead of inheriting its fixed 0.18 thickness.
    tex_v = 0.0f;
#endif
    float4 params = tex2D<float4>(lut_physics, tex_u, tex_v);

    // 物理映射完全保留 (绝热锁死, 不动)
    float density   = params.x * __expf(noise);
    float temp      = params.y * __expf(noise * 0.67f);
    float intensity = params.z * __expf(noise * 2.67f);

#ifdef DISK_CLOUD_STRUCTURE
    // Stable softplus replaces the clipped body boundary with a long density
    // shoulder. Its derivative is continuous through the visible cloud edge.
    float soft_body = fmaxf(body, 0.0f) + edge_softness *
                      log1pf(__expf(-fabsf(body) / edge_softness));
    float cloud_density = powf(soft_body / 0.6f, 1.4f);
    // Soft edges alone leave a nonzero density shoulder everywhere. Along a
    // grazing ray those shoulders add up to a solid sheet. Open correlated
    // 3D channels through the body, retaining dense lobes beside thin mist.
    // Use only two broad octaves so the gaps survive texture filtering.
    float3 gap_pos = make_float3(unwarped.x * 1.7f, unwarped.y * 1.7f,
                                unwarped.z * 2.0f);
    float gap_field = 0.75f * cudaNoise::simplexNoise(gap_pos, 1.0f, npgs_seed + 503) +
                      0.25f * cudaNoise::simplexNoise(gap_pos, 2.0f, npgs_seed + 607);
    float filled = __saturatef((gap_field + 0.25f) / 0.6f);
    filled = filled * filled * (3.0f - 2.0f * filled);
    // Leave the hot inner band continuous; open more gaps outside it.
    float gap_radial = __saturatef((r_disk - 7.0f) / 7.0f);
    gap_radial = gap_radial * gap_radial * (3.0f - 2.0f * gap_radial);
    cloud_density *= 1.0f - porosity * gap_radial * (1.0f - filled * filled);
    // Low-frequency ridges form connected, uneven wisps between the lobes.
    // This is a density component, so it both emits and partially obscures.
    float3 link_pos = make_float3(0.13f * r_disk * cosf(flow_angle),
                                 0.13f * r_disk * sinf(flow_angle),
                                 0.9f * (z_abs - midplane));
    float link_field = cloud_fbm(link_pos, npgs_seed + 419);
    float links = link_density * __expf(-10.0f * fabsf(link_field)) *
                  (0.5f + 0.5f * __saturatef(0.5f + edge_noise));
    cloud_density += links;
    // Fade well before the box boundary, rather than slicing off a plume.
    float edge = __saturatef((z_max - fabsf(z_abs)) / (0.40f * z_max));
    edge = edge * edge * (3.0f - 2.0f * edge);
    float radial_edge = __saturatef((r_max - r_disk) / 7.0f);
    radial_edge = radial_edge * radial_edge * (3.0f - 2.0f * radial_edge);
    density *= cloud_density * __expf(-0.25f * q * q) * edge * radial_edge *
               (1.0f + cold_density_boost * cold_fraction);
    // Keep the original physics LUT temperature and color LUT unchanged.
    // Only brightness changes here; the outer red / inner white mapping stays.
    // The transfer equation already weights emission by density, including
    // its height envelope. Applying a second envelope would thin the light layer.
    intensity *= emission_gain * (1.0f + cold_source_boost * cold_fraction) *
                 __expf(-cold_emission_suppression * cold_fraction);
#endif

    // 末端安全 clip: 直接限制三个输出值, 保证写入 half 纹理 (上限 ~65504) 不溢出.
    // 这道兜底覆盖 noise_clip 管不到的情况 —— params.z 基线大时乘积仍可能爆.
    //   fminf(x, output_clip) 把任何 > output_clip 的值截到 output_clip.
    //   下限 0: density/temp/intensity 物理上非负, 负值无意义.
    density   = fmaxf(0.0f, fminf(density,   output_clip));
    temp      = fmaxf(0.0f, fminf(temp,      output_clip));
    intensity = fmaxf(0.0f, fminf(intensity, output_clip));

    int out_off = (r_idx * z_out * phi_out + z_idx * phi_out + phi_idx) * 4;
    output[out_off + 0] = density;
    output[out_off + 1] = temp;
    output[out_off + 2] = intensity;
#ifdef DISK_CLOUD_STRUCTURE
    output[out_off + 3] = cold_fraction;
#else
    output[out_off + 3] = 0.0f;
#endif
}
'''


def main(output_path=None):
    if not 0.0 <= COLD_CORE_START < COLD_CORE_END <= 1.0:
        raise ValueError("Cold core thresholds must satisfy 0 <= start < end <= 1")
    if min(CLOUD_HEIGHT_INNER, CLOUD_HEIGHT_OUTER) <= 0.0:
        raise ValueError("Cloud heights must be positive")
    if not 0.0 < CLOUD_OUTER_COVERAGE <= 1.05:
        raise ValueError("Outer coverage must be in (0, 1.05]")
    if CLOUD_EDGE_SOFTNESS <= 0.0:
        raise ValueError("Cloud edge softness must be positive")
    if not 0.0 <= CLOUD_POROSITY <= 1.0:
        raise ValueError("Cloud porosity must be in [0, 1]")
    if min(CLOUD_EDGE_ROUGHNESS, CLOUD_LINK_DENSITY, CLOUD_MIDPLANE_WARP,
           COLD_DENSITY_BOOST, COLD_EMISSION_SUPPRESSION,
           CLOUD_EMISSION_GAIN, COLD_SOURCE_BOOST) < 0.0:
        raise ValueError("Cloud appearance controls must be nonnegative")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. 加载 lut_physics 并创建纹理
    print("加载 cache/disk_lut_for_mov_disk.npy ...")
    lut = np.load(os.path.join(base_path, 'cache', 'disk_lut_for_mov_disk.npy')).astype(np.float32)
    print(f"  尺寸: {lut.shape}")

    lut_gpu = cp.asarray(lut)
    tex_lut = create_2d_texture(lut_gpu)
    print("  lut_physics 纹理对象已创建")

    # 2. 分配输出
    total_voxels = R_SAMPLES * Z_SAMPLES * PHI_SAMPLES
    output_flat = cp.empty((total_voxels * 4,), dtype=cp.float32)

    print(f"\n预烘焙 3D 噪声纹理 ({'胞状厚云团' if CLOUD_STRUCTURE else 'NPGS 风格'})...")
    print(f"  分辨率: {R_SAMPLES} (r) x {Z_SAMPLES} (z) x {PHI_SAMPLES} (phi)")
    print(f"  总像素: {total_voxels:,} ({total_voxels * 4 * 4 / 1024 / 1024:.1f} MB)")
    if CLOUD_STRUCTURE:
        print(f"  Cloud coverage={CLOUD_OUTER_COVERAGE}, roughness={CLOUD_EDGE_ROUGHNESS}, emission={CLOUD_EMISSION_GAIN}")
        print(f"  Porosity={CLOUD_POROSITY}, edge softness={CLOUD_EDGE_SOFTNESS}, links={CLOUD_LINK_DENSITY}")
        print("  保留原物理 LUT 色温及颜色映射")
    print(f"  Noise 参数 (cloud 模式仅使用限频 fBm):")
    print(f"    octave [{NPGS_OCTAVE_START}, {NPGS_OCTAVE_END}], contrast={NPGS_CONTRAST}")
    print(f"    center_ref={NPGS_CENTER_REFERENCE}, center_target={NPGS_CENTER_TARGET}, amplitude={NPGS_AMPLITUDE}")
    print(f"    noise_clip={NPGS_NOISE_CLIP} (exp(clip*2.67)={np.exp(NPGS_NOISE_CLIP*2.67):.2e})")
    print(f"    output_clip={OUTPUT_CLIP} (half-safe, 末端硬限幅)")
    print(f"    seed={NPGS_SEED} (0 = 原始盘; 非0 = 换一张盘)")

    # 3. 编译并运行
    options = [f'-I{os.path.join(base_path, "krnls")}']
    if CLOUD_STRUCTURE:
        options.append('-DDISK_CLOUD_STRUCTURE')
    kernel = cp.RawKernel(
        KERNEL_CODE, 'prebake_disk_kernel',
        options=tuple(options)
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
            np.float32(NPGS_AMPLITUDE * (CLOUD_DETAIL_STRENGTH if CLOUD_STRUCTURE else 1.0)),
            np.float32(NPGS_NOISE_CLIP),
            np.float32(OUTPUT_CLIP),
            np.int32(NPGS_SEED),
            np.float32(CLOUD_HEIGHT_INNER), np.float32(CLOUD_HEIGHT_OUTER),
            np.float32(CLOUD_HEIGHT_VARIATION),
            np.float32(CLOUD_EDGE_ROUGHNESS), np.float32(CLOUD_EDGE_SOFTNESS),
            np.float32(CLOUD_LINK_DENSITY), np.float32(CLOUD_POROSITY),
            np.float32(CLOUD_MIDPLANE_WARP),
            np.float32(CLOUD_OUTER_COVERAGE),
            np.float32(CLOUD_EMISSION_GAIN), np.float32(COLD_SOURCE_BOOST),
            np.float32(COLD_DENSITY_BOOST),
            np.float32(COLD_EMISSION_SUPPRESSION),
            np.float32(COLD_CORE_START), np.float32(COLD_CORE_END),
        )
    )
    cp.cuda.Device().synchronize()

    # 4. 保存
    result = output_flat.get().reshape(R_SAMPLES, Z_SAMPLES, PHI_SAMPLES, 4)
    cache_dir = os.path.join(base_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    filename = 'prebaked_disk_clouds.npy' if CLOUD_STRUCTURE else 'prebaked_disk_noise_npgs.npy'
    out_path = os.fspath(output_path) if output_path is not None else os.path.join(cache_dir, filename)
    np.save(out_path, result)

    print(f"\n已保存到 {out_path}")
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

    print(f"\n超过 65000 的像素占比:")
    print(f"  density:   {(result[..., 0] > OUTPUT_CLIP).sum() / result[..., 0].size * 100:.8f}%")
    print(f"  temp:      {(result[..., 1] > OUTPUT_CLIP).sum() / result[..., 1].size * 100:.8f}%")
    print(f"  intensity: {(result[..., 2] > OUTPUT_CLIP).sum() / result[..., 2].size * 100:.8f}%")
    print(f"\n超过 65000 的最大值:")
    print(f"  density:   {result[result[..., 0] > OUTPUT_CLIP, 0].max() if (result[..., 0] > OUTPUT_CLIP).any() else 0:.2e}")
    print(f"  temp:      {result[result[..., 1] > OUTPUT_CLIP, 1].max() if (result[..., 1] > OUTPUT_CLIP).any() else 0:.2e}")
    print(f"  intensity: {result[result[..., 2] > OUTPUT_CLIP, 2].max() if (result[..., 2] > OUTPUT_CLIP).any() else 0:.2e}")
    return result
if __name__ == '__main__':
    main()
