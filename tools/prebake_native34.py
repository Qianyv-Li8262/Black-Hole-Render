# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Bake native noise + separate j/alpha + flared envelope + temperature noise.

RGBA = (fluid extinction, temperature, intrinsic emission strength, reserved).
No experimental modules are imported. J/A mapping is adapted from NPGS,
GPL-3.0, commit 91305ca1661f18b60d6d33ed4616e4cd5b977b9f.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTING_KEYS = (
    'opacity_inner_multiplier', 'opacity_outer_multiplier',
    'opacity_transition_start', 'opacity_transition_end',
    'height_inner', 'height_outer', 'height_variation', 'midplane_warp',
    'temperature_noise_strength',
    'emission_noise_power', 'extinction_noise_power',
    'noise_octave_start', 'noise_roughness',
)

SOURCE=r'''
#include "cuda_noise.cuh"
__device__ float smooth_unit(float t) {
    t = __saturatef(t);
    return t*t*(3-2*t);
}
#ifdef DISK_FILTERED_NOISE
__device__ float filtered_noise(float3 p,float footprint,int start,float roughness) {
    float acc=10.0f;
    for(int octave=start;octave<11;++octave) {
        float frequency=powf(3.0f,(float)octave);
        // Conservative voxel-footprint cutoff, not an exact spectral filter:
        // smooth value noise and nonlinear coefficient maps are not bandlimited.
        float weight=powf(roughness,(float)(octave-start))*
                     smooth_unit((0.5f-frequency*footprint)/0.35f);
        if(weight<=0) break;
        float3 q=make_float3(p.x*frequency,p.y*frequency,p.z*frequency);
        acc*=1+0.1f*weight*cudaNoise::npgsValueNoise3D(q,1919810);
    }
    float x=80*logf(acc*0.1f);
    return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x)));
}
#endif
extern "C" __global__ void bake(cudaTextureObject_t physical,const float* core,
    const float* settings,float4* out,float4* reference,int nr,int nz,int np){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr*nz*np)return;
    int ip=i%np,iz=(i/np)%nz,ir=i/(np*nz);
    float r=4.9495f+30.0505f*(ir+.5f)/nr;
    float z=((iz+.5f)/nz-.5f)*5.0f,phi=(ip+.5f)/np*6.283185307f;
    float4 radial=tex2D<float4>(physical,(r-4.9495f)/30.0505f,0.0f);
    float3 p=make_float3(.1f*r,.1f*z,.02f*powf(r,.7f)*phi);
    float Q=cudaNoise::npgsAccretionDiskNoise(p,2,11,80,1919810);
    float n=cudaNoise::npgsAccretionDiskNoiseCentered(p,2,11,80,3.5f,.1f,.2f,4,1919810)*__saturatef(6.9f-.2f*r);
    float flow=phi+.7f*logf(r/4.9495f);
    float3 hp=make_float3(.28f*r*cosf(flow),.28f*r*sinf(flow),0);
    float H=(settings[4]+(settings[5]-settings[4])*powf((r-4.9495f)/30.0505f,.75f))*
        (1+settings[6]*cudaNoise::simplexNoise(hp,1,1919827));
    H=fmaxf(H,.01f);
    float middle=settings[7]*H*cudaNoise::simplexNoise(hp,1.3f,1919833);
    float q=(z-middle)/H;
    float D=expf(-.25f*q*q)*smooth_unit((2.5f-fabsf(z)))*smooth_unit((35-r)/7);
    // Independent texture contrast: 1 reproduces Q, 0 removes its modulation.
    // Keep exact zeros for positive powers so noise gaps are not filled in.
    float emission_noise=settings[9]==0 ? 1.0f : (settings[9]==1 ? Q : powf(Q,settings[9]));
    float extinction_noise=settings[10]==0 ? 1.0f : (settings[10]==1 ? Q : powf(Q,settings[10]));
    float temperature_noise=expf(settings[8]*n);
#ifdef DISK_FILTERED_NOISE
    float dr=30.0505f/nr,dz=5.0f/nz,dphi=6.283185307f/np;
    float radial_phi=0.014f*powf(r,-0.3f)*phi*dr;
    float footprint=fmaxf(sqrtf(0.01f*dr*dr+radial_phi*radial_phi),
                         fmaxf(0.1f*dz,0.02f*powf(r,0.7f)*dphi));
    emission_noise=extinction_noise=temperature_noise=0;
    // Average nonlinear coefficient modulation over eight subvoxel points;
    // filtering Q first and then raising it to a power biases mean emission.
    for(int sample=0;sample<8;++sample) {
        float sr=r+((sample&1)?0.25f:-0.25f)*dr;
        float sz=z+((sample&2)?0.25f:-0.25f)*dz;
        float sp=phi+((sample&4)?0.25f:-0.25f)*dphi;
        float3 pos=make_float3(0.1f*sr,0.1f*sz,0.02f*powf(sr,0.7f)*sp);
        float value=filtered_noise(pos,footprint,(int)settings[11],settings[12]);
        emission_noise+=0.125f*(settings[9]==0?1:powf(value,settings[9]));
        extinction_noise+=0.125f*(settings[10]==0?1:powf(value,settings[10]));
        float centered=fminf(4.0f,fmaxf(-4.0f,(value/3.5f-0.1f)*0.2f))*
                       __saturatef(6.9f-0.2f*sr);
        temperature_noise+=0.125f*expf(settings[8]*centered);
    }
#endif
    float J=1.4f*D*emission_noise,A=D*D/0.3f*extinction_noise;
    // Editable radial extinction curve: inner -> outer using smoothstep.
    float blend=smooth_unit((r-settings[2])/(settings[3]-settings[2]));
    float multiplier=settings[0]+(settings[1]-settings[0])*blend;
    float kappa=.5f*(.6f+8.4f*core[ir])*multiplier;
    float T=fminf(65000.0f,radial.y*temperature_noise);
    // Store per-length j directly, avoiding a potentially enormous j/sigma
    // source function in the thin tails of D. The renderer integrates j exactly
    // for a constant segment, with a well-defined zero-extinction limit.
    out[i]=make_float4(radial.x*A*kappa,T,1.7f*radial.x*radial.z*J,0);
#ifdef DISK_FILTERED_NOISE
    bool need_reference=true;
#else
    bool need_reference=settings[9]!=1 || settings[10]!=1;
#endif
    if(need_reference)
        reference[i]=make_float4(radial.x*D*D/0.3f*Q*kappa,
                                fminf(65000.0f,radial.y*expf(settings[8]*n)),
                                1.7f*radial.x*radial.z*1.4f*D*Q,0);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=ROOT / 'cfg/disk_material.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'cache/disk_native34_coefficients.npy')
    parser.add_argument('--resolution', type=int, nargs=3, metavar=('R', 'Z', 'PHI'),
                        help='Override bake resolution without editing material settings')
    args = parser.parse_args()
    settings_bytes = args.settings.read_bytes()
    settings = json.loads(settings_bytes)
    settings.setdefault('emission_noise_power', 1.0)
    settings.setdefault('extinction_noise_power', 1.0)
    settings.setdefault('noise_mode', 'legacy')
    settings.setdefault('noise_octave_start', 2)
    settings.setdefault('noise_roughness', 1.0)
    settings.setdefault('noise_filter_mix', 1.0)
    if settings['noise_mode'] not in ('legacy', 'filtered'):
        raise ValueError('noise_mode must be legacy or filtered')
    filtered_noise = settings['noise_mode'] == 'filtered'
    if (type(settings['noise_octave_start']) is not int or not 0 <= settings['noise_octave_start'] < 11
            or not 0 <= settings['noise_roughness'] <= 1):
        raise ValueError('noise_octave_start must be an integer in [0, 10]; noise_roughness must be in [0, 1]')
    if not 0 <= settings['noise_filter_mix'] <= 1:
        raise ValueError('noise_filter_mix must be in [0, 1]')
    if args.resolution is not None:
        settings['resolution'] = args.resolution
    settings_bytes = json.dumps(settings, sort_keys=True).encode('utf-8')
    resolution = settings['resolution']
    if len(resolution) != 3 or any(type(x) is not int or x < 2 for x in resolution):
        raise ValueError('resolution must contain three integer sizes >= 2')
    nr, nz, nphi = resolution
    values = np.array([settings[k] for k in SETTING_KEYS], np.float32)
    if (not np.isfinite(values).all() or min(values[:2]) <= 0
            or not 0 < values[2] < values[3]
            or min(values[4:6]) <= 0 or min(values[6:]) < 0):
        raise ValueError('Invalid material settings: positive opacity/height, start < end, nonnegative noise strengths required')
    if args.output.suffix != '.npy':
        raise ValueError('--output must end in .npy')
    physical_path = ROOT / 'cache/disk_lut_for_mov_disk.npy'
    if not physical_path.is_file():
        raise FileNotFoundError('Generate the radial physics LUT first: python tools/generate_lut_physics.py')

    scratch = ROOT / 'temp/native34_bake'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(scratch)
    tempfile.tempdir = str(scratch)
    import cupy as cp
    sys.path.insert(0, str(ROOT))
    from helpers.cuda_tex import create_texture_array_2d

    physical = np.load(physical_path)
    texture = create_texture_array_2d(cp.asarray(physical, np.float32), 4, (1, 1, 1, 1))
    core_path = ROOT / 'tools/data/disk_core_profile.json'
    profile_bytes = core_path.read_bytes()
    profile = np.array(json.loads(profile_bytes), np.float32)
    radii = 4.9495 + 30.0505 * (np.arange(nr) + .5) / nr
    core = np.interp(radii, profile[:, 0], profile[:, 1]).astype(np.float32)
    core_gpu, values_gpu = cp.asarray(core), cp.asarray(values)

    def bake_coefficients(filtered):
        out = cp.empty((nr, nz, nphi, 4), cp.float32)
        adjusted = filtered or values[9] != 1 or values[10] != 1
        reference = cp.empty((nr, nz, nphi, 4) if adjusted else (1,), cp.float32)
        options = [f'-I{ROOT / "krnls"}']
        if filtered:
            options.append('-DDISK_FILTERED_NOISE')
        cp.RawKernel(SOURCE, 'bake', options=tuple(options))(
            ((nr*nz*nphi + 255)//256,), (256,),
            (np.uint64(texture.ptr), core_gpu, values_gpu, out, reference,
             np.int32(nr), np.int32(nz), np.int32(nphi)))
        if adjusted:
            # Preserve each shell's j/alpha means and, in filtered mode, its
            # temperature mean. This does not guarantee identical pixel colors:
            # transfer, Doppler shifting, and tone mapping remain nonlinear.
            for channel, power in ((0, values[10]), (1, 1), (2, values[9])):
                if not filtered and power == 1:
                    continue
                target = reference[..., channel].sum(axis=(1, 2), dtype=cp.float64)
                total = out[..., channel].sum(axis=(1, 2), dtype=cp.float64)
                scale = target / cp.maximum(total, cp.finfo(cp.float64).tiny)
                out[..., channel] *= scale.astype(cp.float32)[:, None, None]
        return out

    mix = settings['noise_filter_mix'] if filtered_noise else 0.0
    out = bake_coefficients(filtered_noise and mix > 0)
    if filtered_noise and 0 < mix < 1:
        # Camera-independent blend of normalized coefficients. This deliberately
        # retains some legacy grain/aliasing as an appearance-preserving option.
        legacy = bake_coefficients(False)
        out *= mix
        out += (1-mix)*legacy
    result = out.get()
    if not np.isfinite(result).all() or (result[..., :3] < 0).any():
        raise RuntimeError('Invalid material bake')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    metadata = dict(
        format='native34-coefficients-v1',
        channels=['extinction', 'temperature', 'emission', 'reserved'],
        dtype='float32', settings=settings,
        noise_normalization='per-radius-coefficient-sum-v1',
        noise_filter='voxel-footprint-fade-and-8-point-coefficient-average-v1' if filtered_noise else 'none',
        settings_sha256=hashlib.sha256(settings_bytes).hexdigest(),
        core_profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        physics_lut_sha256=hashlib.sha256(physical_path.read_bytes()).hexdigest(),
    )
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved', args.output, result.shape, 'ranges',
          result.min(axis=(0, 1, 2)), result.max(axis=(0, 1, 2)))


if __name__ == '__main__':
    main()
