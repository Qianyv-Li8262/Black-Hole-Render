// Pure scalar reference implementation of the black-hole ray kernel.

#include "cpu_render_common.h"

void render_one_tile_scalar(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                            const render_params &params, int tile_width, int tile_height, int tile_x, int tile_y)
{
    const int x_begin = tile_x * tile_width;
    const int y_begin = tile_y * tile_height;
    const int x_end = std::min(x_begin + tile_width, params.imgwidth);
    const int y_end = std::min(y_begin + tile_height, params.imgheight);

    const float3 fwd = make_float3(params.fwd_x, params.fwd_y, params.fwd_z);
    const float3 right = make_float3(params.right_x, params.right_y, params.right_z);
    const float3 up = make_float3(params.up_x, params.up_y, params.up_z);
    const float3 beta = make_float3(params.vfwd, params.vright, params.vup);
    const float gamma = rsqrtf_cpu(1.0f - beta * beta);
    const float3 initial_cam_pos = make_float3(params.cam_pos_x, params.cam_pos_y, params.cam_pos_z);
    const float camera_radius = length(initial_cam_pos);
    const float initial_u = 1.0f / (2.0f * camera_radius);
    const float initial_upl = 1.0f + initial_u;
    const float3 beta_global = beta.x * fwd + beta.y * right + beta.z * up;
    const float3 e0 = beta_global * gamma / (initial_upl * initial_upl);
    float3 e1_fwd = boost(beta, make_float3(1.0f, 0.0f, 0.0f), gamma);
    e1_fwd = (e1_fwd.x * fwd + e1_fwd.y * right + e1_fwd.z * up) / (initial_upl * initial_upl);
    float3 e2_right = boost(beta, make_float3(0.0f, 1.0f, 0.0f), gamma);
    e2_right = (e2_right.x * fwd + e2_right.y * right + e2_right.z * up) / (initial_upl * initial_upl);
    float3 e3_up = boost(beta, make_float3(0.0f, 0.0f, 1.0f), gamma);
    e3_up = (e3_up.x * fwd + e3_up.y * right + e3_up.z * up) / (initial_upl * initial_upl);

    for (int pixel_y = y_begin; pixel_y < y_end; ++pixel_y) {
        for (int pixel_x = x_begin; pixel_x < x_end; ++pixel_x) {
            float4 buffer = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            for (int sample = 0; sample < params.jitternum; ++sample) {
                // The CUDA source currently passes 1 rather than `frames` here.
                const float2 jitter = hammersley(sample, params.jitternum, static_cast<std::uint32_t>(pixel_x),
                                                 static_cast<std::uint32_t>(pixel_y), 1u);
                const float physical_x =
                    ((static_cast<float>(pixel_x) + jitter.x) / static_cast<float>(params.imgwidth) - 0.5f) *
                    params.physwidth;
                const float physical_y =
                    ((static_cast<float>(pixel_y) + jitter.y) / static_cast<float>(params.imgheight) - 0.5f) *
                    params.physheight;
                float3 cam_pos = initial_cam_pos;
                float delta_t = 0.0f;
                const float3 camera_ray = normalize(make_float3(params.focal_length, physical_x, -physical_y));

                float radius = length(cam_pos);
                float u = 1.0f / (2.0f * radius);
                float upl = 1.0f + u;
                float umi = 1.0f - u;
                float factor = upl / umi;
                float n = upl * upl * upl / umi;
                float3 direction =
                    normalize(camera_ray.x * e1_fwd + camera_ray.y * e2_right + camera_ray.z * e3_up - e0);
                if (has_flag(params, DEPTH_JITTER)) {
                    const std::uint32_t depth_seed =
                        pcg_hash(static_cast<std::uint32_t>(pixel_x) ^
                                 pcg_hash(static_cast<std::uint32_t>(pixel_y) ^
                                          pcg_hash(static_cast<std::uint32_t>(sample) ^
                                                   pcg_hash(static_cast<std::uint32_t>(params.frames)))));
                    const float depth_jitter = static_cast<float>(depth_seed) / 4294967296.0f;
                    cam_pos =
                        cam_pos + direction * (depth_jitter * std::fmax(radius - 1.5f, 0.0f) / 10.0f * params.step);
                }
                float3 p = direction * n;
                const float3 p_init = p;
                const float lz = cam_pos.x * p.y - cam_pos.y * p.x;
                bool trace = true;
                float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                for (int step_index = 0; step_index < params.maxstep && trace; ++step_index) {
                    const float3 prev_pos = cam_pos;
                    const float prev_dt = delta_t;

                    if (has_flag(params, USE_RK4)) {
                        float rmhalf = radius - 0.5f;
                        float gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        float uplsq = upl * upl;
                        float uu = 1.0f / (uplsq * uplsq);
                        const float3 k11 = p * uu;
                        const float3 k12 = gravity * cam_pos;
                        const float k_t1 = uplsq / (umi * umi);
                        const bool in_disk_volume = radius > 4.5f && radius < 27.0f && std::fabs(cam_pos.z) < 3.0f;
                        const float zone_multiplier =
                            in_disk_volume ? 0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f) : 1.0f;
                        float current_step =
                            params.step * std::fmin(50.0f, std::fmax(0.005f, radius - 0.54f)) * zone_multiplier * 5.0f;
                        if (has_flag(params, PHOTON_RING_OPTIMIZATION)) {
                            const float dist_to_ps = std::fabs(radius - 1.866025f);
                            current_step *= 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
                        }

                        float step_half = current_step * 0.5f;
                        float3 pos_tmp = cam_pos + step_half * k11;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k21 = (p + step_half * k12) * uu;
                        const float3 k22 = pos_tmp * gravity;
                        const float k_t2 = uplsq / (umi * umi);

                        pos_tmp = cam_pos + step_half * k21;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k31 = (p + step_half * k22) * uu;
                        const float3 k32 = pos_tmp * gravity;
                        const float k_t3 = uplsq / (umi * umi);

                        pos_tmp = cam_pos + current_step * k31;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k41 = (p + current_step * k32) * uu;
                        const float3 k42 = pos_tmp * gravity;
                        const float k_t4 = uplsq / (umi * umi);

                        step_half = current_step * 0.16666666667f;
                        cam_pos = cam_pos + step_half * (k11 + k41 + 2.0f * k21 + 2.0f * k31);
                        p = p + step_half * (k12 + k42 + 2.0f * k22 + 2.0f * k32);
                        delta_t += step_half * (k_t1 + k_t4 + 2.0f * k_t2 + 2.0f * k_t3);
                        radius = length(cam_pos);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        n = upl * upl * upl / umi;
                        p = normalize(p) * n;
                    } else {
                        float rmhalf = radius - 0.5f;
                        float gravity = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
                        float uplsq = upl * upl;
                        float uu = 1.0f / (uplsq * uplsq);
                        const float3 k11 = p * uu;
                        const float3 k12 = gravity * cam_pos;
                        const bool in_disk_volume = radius > 4.5f && radius < 37.0f && std::fabs(cam_pos.z) < 3.0f;
                        const float zone_multiplier =
                            in_disk_volume ? 0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f) : 1.0f;
                        float current_step =
                            params.step * std::fmin(50.0f, std::fmax(0.005f, radius - 0.54f)) * zone_multiplier;
                        if (has_flag(params, PHOTON_RING_OPTIMIZATION)) {
                            const float dist_to_ps = std::fabs(radius - 1.866025f);
                            current_step *= 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
                        }
                        const float step_half = current_step * 0.5f;
                        const float3 pos_tmp = cam_pos + step_half * k11;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k21 = (p + step_half * k12) * uu;
                        const float3 k22 = pos_tmp * gravity;
                        const float k_t2 = uplsq / (umi * umi);
                        cam_pos = cam_pos + current_step * k21;
                        p = p + current_step * k22;
                        delta_t += current_step * k_t2;
                        radius = length(cam_pos);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        n = upl * upl * upl / umi;
                        p = normalize(p) * n;
                    }

                    float3 disk_pos;
                    float disk_time;
                    if (has_flag(params, RAND_SAMP_DISK)) {
                        const float random_sample =
                            rand_float(pcg_hash(static_cast<std::uint32_t>(pixel_x) ^
                                                pcg_hash(static_cast<std::uint32_t>(pixel_y) ^
                                                         pcg_hash(static_cast<std::uint32_t>(step_index) ^
                                                                  pcg_hash(static_cast<std::uint32_t>(sample))))));
                        disk_pos = cam_pos * random_sample + prev_pos * (1.0f - random_sample);
                        disk_time = delta_t * random_sample + prev_dt * (1.0f - random_sample);
                    } else {
                        disk_pos = (cam_pos + prev_pos) * 0.5f;
                        disk_time = (prev_dt + delta_t) * 0.5f;
                    }

                    const float disk_radius_sq = disk_pos.x * disk_pos.x + disk_pos.y * disk_pos.y;
                    const bool in_disk =
                        disk_radius_sq > 24.4974f && disk_radius_sq < 1225.0f && std::fabs(cam_pos.z) < 2.5f;
                    if (in_disk) {
                        const float disk_radius = std::sqrt(disk_radius_sq);
                        float td = 0.0f;
                        float pd = 0.0f;
                        tdpd(disk_radius, &td, &pd);
                        const float rotation = pd * (params.time - disk_time) / td;
                        const float phi_final = fast_mod2pi(std::atan2(disk_pos.y, disk_pos.x) + rotation);
                        const float4 disk_parameters =
                            disk.sample(phi_final * 0.15915494f, (disk_pos.z / 2.5f) / 2.0f + 0.5f,
                                        (disk_radius - 4.9495f) / 30.0505f);
                        const float g = std::fmax(
                            (std::fabs((factor * gamma + p_init * e0) / (td - pd * lz)) - 1.0f) + 1.0f, 0.01f);
                        const float ravg2 = length(prev_pos) + radius;
                        const float uuu = 1.0f + 1.0f / ravg2;
                        const float g4 = g * g * g * g;
                        const float step_len = length(cam_pos - prev_pos);
                        const float kzg4 = 2.0f * disk_parameters.z;
                        float step_opacity;
                        if (has_flag(params, OPACITY_CHANGE)) {
                            const float temp_eff = disk_parameters.y * g;
                            const float cold_factor = 1.0f + 2.0f * saturate((5000.0f - temp_eff) / 3000.0f);
                            step_opacity = disk_parameters.x * 3.0f * uuu * uuu *
                                           std::fma(step_len, -std::exp(-kzg4 * kzg4), step_len) / g * cold_factor;
                        } else {
                            step_opacity = disk_parameters.x * 1.7f * uuu * uuu *
                                           std::fma(step_len, -std::exp(-kzg4 * kzg4), step_len) / g;
                        }
                        const float temp_exp = -std::exp(-step_opacity);
                        float4 emission;
                        if (has_flag(params, DISK_DOPPLER_FOLLOW_BACKGROUND)) {
                            float4 color = color_lut.sample((disk_parameters.y - 510.0f) / 20000.0f, 0.5f);
                            const float3 shifted_rgb =
                                rgb_three_line_frequency_shift(make_float3(color.x, color.y, color.z), g);
                            emission = make_float4(shifted_rgb.x * disk_parameters.z * g4,
                                                   shifted_rgb.y * disk_parameters.z * g4,
                                                   shifted_rgb.z * disk_parameters.z * g4, 1.0f);
                        } else {
                            emission = disk_emission_dep(std::fmax(disk_parameters.y * g, 1000.0f),
                                                         disk_parameters.z * g4, color_lut);
                        }

                        float temp_calc = std::fma(emission.x, -accumulated_color.w, emission.x);
                        accumulated_color.x += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = std::fma(emission.y, -accumulated_color.w, emission.y);
                        accumulated_color.y += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = std::fma(emission.z, -accumulated_color.w, emission.z);
                        accumulated_color.z += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = 1.0f - accumulated_color.w;
                        accumulated_color.w += std::fma(temp_calc, temp_exp, temp_calc);
                        if (accumulated_color.w > 0.99f)
                            trace = false;
                    }

                    if (radius < 0.55f || radius > 140.0f)
                        trace = false;
                }

                float4 color;
                if (radius >= 0.55f && !std::isnan(radius)) {
                    const float3 final_dir = normalize(p);
                    const float phi = std::atan2(final_dir.y, -final_dir.x);
                    const float theta = std::asin(-final_dir.z);
                    float4 bkgd = background.sample(phi * 0.1591549f + 0.5f, theta * 0.3183099f + 0.5f);
                    if (has_flag(params, BACKGROUND_DOPPLER)) {
                        float g_sky = std::fabs(factor * gamma + p_init * e0);
                        g_sky = std::fmin(std::fmax(g_sky, 1e-4f), 20.0f);
                        const float3 shifted_rgb =
                            rgb_three_line_frequency_shift(make_float3(bkgd.x, bkgd.y, bkgd.z), g_sky);
                        bkgd.x = shifted_rgb.x;
                        bkgd.y = shifted_rgb.y;
                        bkgd.z = shifted_rgb.z;
                    }
                    color = accumulated_color + bkgd * (1.0f - accumulated_color.w);
                } else {
                    color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
                }
                buffer = buffer + color;
            }

            buffer = buffer * (1.0f / static_cast<float>(params.jitternum));
            float *output = raw + (static_cast<std::size_t>(pixel_y) * params.imgwidth + pixel_x) * 4;
            output[0] = buffer.x;
            output[1] = buffer.y;
            output[2] = buffer.z;
            output[3] = 0.0f;
        }
    }
}
