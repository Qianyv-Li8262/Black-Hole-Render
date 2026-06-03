import numpy as np
import cupy as cp
import cv2
import time
import threading
import os
import sys
from cuda_tex import create_texture_array_2d


# ---------- 物理计算函数(与原代码一致) ----------
def update_camera_physics_analytical(tau, r_start, dir_unit, fwd, right, up, d_tau=1.0):
    """
    计算从无穷远处静止释放的相机,在下落到本征时间 tau 时的严格解析物理状态。
    """
    M = 1.0
    R_start = r_start + 1.0 + 0.25 / r_start
    tau_max = (np.sqrt(2.0) / 3.0) * (R_start ** 1.5)

    if tau >= tau_max:
        r_next = 0.51
        beta_mag = 0.999
    else:
        R = (R_start ** 1.5 - (1.5 * np.sqrt(2.0)) * tau) ** (2.0 / 3.0)
        R = max(2.0001, R)
        r_next = 0.5 * ((R - 1.0) + np.sqrt((R - 1.0) ** 2 - 1.0))
        u_next = 1.0 / (2.0 * r_next)
        beta_mag = np.sqrt(2.0 / r_next) / (1.0 + u_next)
        beta_mag = min(0.999, beta_mag)

    beta_global = -beta_mag * dir_unit
    vx_next = np.dot(beta_global, fwd)
    vy_next = np.dot(beta_global, right)
    vz_next = np.dot(beta_global, up)

    u_next = 1.0 / (2.0 * r_next)
    factor = (1.0 + u_next) / (1.0 - u_next)
    gamma = 1.0 / np.sqrt(1.0 - beta_mag ** 2)
    dt = factor * gamma * d_tau

    return r_next, vx_next, vy_next, vz_next, dt


# ---------- 主程序 ----------
if __name__ == '__main__':
    # ------------------- 纹理与核函数初始化 -------------------
    base_path = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.join(base_path, 'starmap_random_2020_16k.exr')
    print('Start texture creating.')
    img_bgr = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)

    if img_bgr is None:
        print(f"错误:无法在路径 {img_file_path} 找到背景图片!")
        exit()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) * 300
    del img_bgr
    img_float = cp.asarray(img_rgb.astype(np.float16))
    del img_rgb
    skybox_rgba = cp.zeros((*img_float.shape[:2], 4), dtype=cp.float16)
    skybox_rgba[:, :, :3] = img_float
    del img_float
    tex_handle = create_texture_array_2d(skybox_rgba, 4, (1, 1, 1, 1), is_half=True)
    del skybox_rgba
    print('Texture created successfully, starting kernel compilation and LUT initializing.')

    physlut_file_path = os.path.join(base_path, 'disk_lut_for_mov_disk.npy')
    lut_phys = cp.load(physlut_file_path).astype(cp.float32)
    tex_handle_lut = create_texture_array_2d(lut_phys, 4, (1, 1, 1, 1))

    colorlut_file_path = os.path.join(base_path, 'color_lut2.npy')
    lut_color = cp.load(colorlut_file_path).astype(cp.float32)
    lut_rgba = cp.zeros((*lut_color.shape[:2], 4), dtype=cp.float32)
    lut_rgba[:, :, :3] = lut_color
    tex_handle_color = create_texture_array_2d(lut_rgba, 4, (1, 1, 1, 1))

    kernel_path = os.path.join(base_path, "blackholekernel3_moving disk speed.cu")
    with open(kernel_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()
    include_dir = os.path.dirname(os.path.abspath(kernel_path))
    module = cp.RawModule(code=cuda_source, options=('-use_fast_math', "-I", include_dir))
    trace_rays_kernel = module.get_function("blackholekernel")

    kernel_path = os.path.join(base_path, "postprocess_gemini.cu")
    with open(kernel_path, "r", encoding="utf-8") as f:
        cuda_source = f.read()
    bloom_module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
    extract_bright_kernel = bloom_module.get_function("extract_bright_kernel")
    blur_x_kernel = bloom_module.get_function("blur_x_kernel")
    blur_y_fuse_kernel = bloom_module.get_function("blur_y_fuse_postprocess_kernel")

    print('Kernels compiled.')

    # ------------------- 参数配置 -------------------
    w, h = 3200, 2000
    total_frames = 1800
    start_t = 75
    SSAA_COUNT = 128
    output_dir = os.path.join(base_path, 'output_frames')
    os.makedirs(output_dir, exist_ok=True)

    # 缓存分配
    frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
    bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
    blur_x_tmp = cp.empty((h * w * 4), dtype=cp.float32)
    current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

    bloom_threshold = np.float32(1.7)
    bloom_radius = np.int32(20)
    bloom_sigma = np.float32(8.0)
    bloom_strength = np.float32(1.5)

    block_x, block_y = 32, 8
    grid_x = w // block_x + 1 if w % block_x != 0 else w // block_x
    grid_y = h // block_y + 1 if h % block_y != 0 else h // block_y

    # 相机初始状态
    cam_pos_init = np.array([10.0, 0.0, 1], dtype=np.float32)
    r0 = np.linalg.norm(cam_pos_init)
    dir_unit = cam_pos_init / r0
    r = r0
    t = start_t
    tau = 0.0
    vx, vy, vz = 0.1, -0.3, 0.0
    d_tau = 1.0

    cam_yaw = -3.14
    cam_pitch = -0.1488899
    cam_roll = 0.333
    focal_length = 1

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
    fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
    fwd_z = np.sin(cam_pitch)
    fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
    fwd /= np.linalg.norm(fwd)
    right0 = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right0)
    if right_norm > 1e-6:
        right0 /= right_norm
    else:
        right0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up0 = np.cross(right0, fwd)
    up0 /= np.linalg.norm(up0)
    right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
    up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)

    cam_pos = r * dir_unit

    # ------------------- 渲染核心函数(在子线程中运行) -------------------
    def render_single_frame(frame_idx, ssaa):
        """
        执行完整的一帧渲染,包括所有kernel启动和同步。
        如果发生设备重置,会抛出异常。
        """
        trace_rays_kernel((grid_x, grid_y), (block_x, block_y),
                          (frame_intermediate_result, cp.uint64(tex_handle.ptr), cp.uint64(tex_handle_lut.ptr),
                           cp.uint64(tex_handle_color.ptr),
                           cp.float32(t),
                           cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
                           cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
                           cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
                           cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
                           cp.float32(vx), cp.float32(vy), cp.float32(vz),
                           cp.int32(w), cp.int32(h),
                           cp.float32(3.2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1),
                           cp.int32(2000), cp.int32(ssaa), cp.int32(frame_idx)))

        extract_bright_kernel((grid_x, grid_y), (block_x, block_y),
                              (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h),
                               np.float32(1), bloom_threshold))

        blur_x_kernel((grid_x, grid_y), (block_x, block_y),
                      (bright_buf, blur_x_tmp, np.int32(w), np.int32(h),
                       bloom_radius, bloom_sigma))

        blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
                           (frame_intermediate_result, blur_x_tmp, current_frame_float,
                            np.int32(w), np.int32(h),
                            bloom_radius, bloom_sigma,
                            np.float32(1), bloom_strength))

        cp.cuda.Device().synchronize()  # 阻塞点,若设备重置会抛异常

    # ------------------- 预渲染:估算单帧用时 -------------------
    print('Starting pre-render for time estimation...')
    try:
        pre_thread = threading.Thread(target=render_single_frame, args=(1, 16))
        pre_thread.start()
        pre_start = time.time()
        while pre_thread.is_alive():
            pre_thread.join(timeout=0.1)
        pre_elapsed = time.time() - pre_start

        estimated_per_frame = pre_elapsed * (SSAA_COUNT / 16)
        total_est = estimated_per_frame * total_frames
        print(f'预渲染完成,预计每帧用时 {estimated_per_frame:.3f} s,'
              f'预计总用时:{total_est:.0f} s '
              f'({total_est // 3600:.0f} hr {((total_est % 3600) // 60):.0f} min {total_est % 60:.1f} secs)')
    except KeyboardInterrupt:
        print("\n中断,GPU计算已停止")
        cp.cuda.runtime.deviceReset()
        sys.exit(0)

    # ------------------- 正式渲染循环(可中断) -------------------
    print(f"\n开始离线渲染,总计 {total_frames} 帧,输出目录: {output_dir}")
    frame_idx = 1
    should_exit = False

    try:
        while frame_idx <= total_frames and not should_exit:
            # 更新相机位置(本帧使用)
            cam_pos = r * dir_unit

            print(f"[Frame {frame_idx:05d}/{total_frames:05d}] 开始渲染...")
            t0 = time.time()

            # 在子线程中执行渲染
            render_thread = threading.Thread(target=render_single_frame, args=(frame_idx, SSAA_COUNT))
            render_thread.start()

            # 主线程等待子线程完成,同时保持对KeyboardInterrupt的响应
            while render_thread.is_alive():
                render_thread.join(timeout=0.1)

            # 如果线程正常完成,进行数据后处理和物理更新
            # (若发生设备重置,子线程异常退出,is_alive会变成False,但随后访问CuPy数组会出错,
            #  所以我们需要捕获可能的异常,但这里我们依赖KeyboardInterrupt来触发设备重置,
            #  并在退出时不做后续处理。为了安全,可以加一个标志或直接break)
            # 我们直接在try块中执行后续操作,如果设备重置了,后续操作会抛出异常,被外层的except捕获
            img_rgba_cp = current_frame_float.reshape((h, w, 4))
            img_rgba_np = cp.asnumpy(img_rgba_cp)
            img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)

            file_name = f"frame_{frame_idx:05d}.png"
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, img_bgr_np)

            elapsed = time.time() - t0
            print(f"[Frame {frame_idx:05d}/{total_frames:05d}] 渲染完成 | 耗时: {elapsed:.3f} 秒 | "
                  f"本征时 tau = {tau:.1f} | 坐标位置 r = {r:.4f} | 坐标时 t = {t:.2f} | "
                  f"局部物理速度 beta = {np.sqrt(vx**2+vy**2+vz**2):.4f}")

            # 更新物理状态(为下一帧准备)
            if frame_idx < total_frames:
                tau += d_tau
                r, vx, vy, vz, dt = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)
                t += dt

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n强制中断,GPU计算已停止.")
        # 重置设备,立即杀死所有正在执行的kernel
        cp.cuda.runtime.deviceReset()
        os._exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        cp.cuda.runtime.deviceReset()
        os._exit(1)