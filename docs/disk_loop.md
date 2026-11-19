# Retarded-time two-texture disk loop

This optional mode is implemented directly in the sampling block of `krnls/disk.cu`.
The ordinary path and kernel argument list remain available when `DISK_LOOP` is absent.
The mode is supported by `render_single.py`, `tools/preview_disk.py`,
`render_realtime.py` and the GPU workers in `render_hybrid.py`, using independent
emission/extinction coefficient caches. CPU rendering does not support the
loop; `render_hybrid.py` rejects `--use-cpu` while a loop is enabled.
The disk renderers and preview tool read `DISK_LOOP_PERIOD` from `cfg/offline.py`.
Set it explicitly to `0.0` for single-texture rendering or a positive value for
a loop. A and B must use matching baked settings, vertical bounds, physics LUT
and core profile, with different noise seeds. The renderer checks each array's
dimensions against its JSON sidecar. Keep both JSON sidecars with their textures.
Interactive camera controls remain available. The time step is 0.5 coordinate M per
displayed frame, so a period of 40M spans 80 frames at a fixed camera.
Its initial camera position, orientation, focal length, velocity and coordinate
time also come from `cfg/offline.py`, along with the skybox, sky exposure and
color LUT. The interactive preview stays 1600 pixels wide, matching the
offline aspect ratio and physical sensor size to preserve the field of view.
After initialization the camera remains under keyboard control.
If the camera speed reaches the light-speed limit, the view stays black until
you reduce or reset the velocity.

For frame-parallel multi-GPU throughput, set `DISK_LOOP_PERIOD>0` and run:

```bash
python render_hybrid.py --gpus all --time-step 0.5
```

Each GPU owns A and B and claims complete frames from the shared scheduler.
Time is derived from the frame index, independently of GPU assignment or
completion order: `t=start_t+(frame_idx-1)*time_step`. The camera and configured
velocity stay fixed. `--time-step` is coordinate M per frame (defaults to
`cfg.d_tau` in this entry point); no camera proper-time conversion is applied.
For period 40M and 80 loop frames, use time step 0.5 and `total_frames=80`.
The duplicate endpoint is omitted. Existing frame PNGs are skipped unless
`--overwrite` is specified; use a new output directory when changing settings.

## Physical clocks and the existing velocity convention

All material times below are Schwarzschild coordinate times in M=c=1 units.
They are neither video seconds nor the observer's or fluid's proper time.
In isotropic coordinates let u=1/(2r), alpha=(1-u)/(1+u), B=(1+u)^2.
The metric is ds²=-alpha²dt²+B²dx². The kernel accumulates the null-ray
coordinate travel time, including gravitational delay, and interpolates that
delay to the SAME quadrature position used to sample the material:

    t_em = t_observer - t_use

Do not replace t_use by Euclidean distance or reduce t_use modulo the cycle
independently of the emission clock. Negative t_em is valid: the periodic
emissivity field is defined for the entire time axis, not initialized at frame 1.
For numerical stability the kernel first removes whole periods from
t_observer-e, then subtracts the full t_use before computing the phase.
This is algebraically equivalent to using t_em-e and improves endpoint agreement.

The existing tdpd(r) gives td and pd, with angular-speed magnitude

    Omega(r) = pd/td = 8 r^(3/2)/(2r+1)^3 = R^(-3/2)
    R = r(1+1/(2r))²

on the equatorial circular-orbit model. Reading texture(phi+Omega*t) moves
features toward decreasing phi. Retain this sign: the traced p points from
camera into the scene, opposite the future photon's spatial momentum. With
lz=(x cross p)_z and this disk rotation, the existing emitter-energy factor
is td-pd*lz. Both populations therefore retain the SAME physical bulk velocity,
Doppler/gravitational shift, and comoving path length. No velocity is inferred
from the wrapped texture clock. The thick-disk extension remains the existing
phenomenological use of the equatorial velocity at cylindrical radius.

## Local turnover

Let T>0 be the lifetime/period, a_ref the preferred texture advection time,
and e the emission-coordinate epoch when population A has maximum weight:

    qa = fract((t_em-e)/T + 1/2)
    qb = fract(qa + 1/2)
    ta = a_ref + T(qa-1/2)
    tb = a_ref + T(qb-1/2)
    wa = (1-cos(2*pi*qa))/2
    wb = 1-wa

Sample A at phi+Omega*ta and B at phi+Omega*tb. Do NOT subtract t_use again
from ta or tb: the phase already contains the delay. Between resets,
dta/dt_em=dtb/dt_em=1, so each pattern advects at the original angular speed.
At each clock reset its population's weight and weight derivative vanish,
hiding the jump. The smooth window removes turnover jumps; ordinary linear
texture filtering still has its usual piecewise-smooth spatial derivatives.

Turnover corresponds to prescribed creation/destruction of emissivity and
extinction patterns. It is not a mass-conserving fluid/MHD solution. A shared
coordinate period also means different proper lifetimes T/td(r) for the fluid.
The scheme bounds additional shear instead of claiming each gas parcel closes
its orbit. Very short lifetimes can look like dissolving clouds; long ones
still expose stretching. Independent B avoids half-period A/B swap symmetry,
but two-layer fading can still change contrast and apparent brightness.

The preferred texture age a_ref is NOT a screenshot's observer timestamp.
An old frame taken at t_observer=20 contains many ages 20-t_use. There is no
single new a_ref that reproduces that entire retarded image exactly.

## Radiative transfer and closure

Convert each population's temperature and emission strength to shifted RGB
emission separately. Then form

    sigma = wa*sigma_A + wb*sigma_B
    j_RGB = wa*j_RGB_A + wb*j_RGB_B

Integrate this one effective medium with the existing transfer integral.
Do not average temperatures, source functions j/sigma, or finished images.
The nonnegative complementary weights preserve coefficient validity, not
necessarily equal rendered brightness at every phase.

At a fixed camera the metric, rays and delays are stationary. Consequently
t_em(t_observer+T)=t_em(t_observer)+T at EVERY sample, including longer winding
paths. Coefficients and the transfer solution therefore repeat after T.
Different disk regions and multiple images can show different turnover phases
within the SAME observed frame; a global full-disk disappearance is unnecessary.
This is periodic in the real-number model, with normal float32 numerical error
in the renderer. Very large absolute scene times eventually lose phase precision.

The loop forces fixed depth sampling so frame-dependent jitter does not break
numerical closure. A static supported camera has dt/dtau_camera=1/alpha_camera;
`--loop-frames N` uses d_tau=alpha_camera*T/N. It saves N frames over [t0,t0+T),
omitting the duplicate endpoint. Choose playback FPS independently: N/FPS is
the video's duration in seconds. Static mode requires zero camera velocity;
it represents an observer held at fixed coordinates, not free fall.

## Try it

Keep the existing A cache. Set the period and B's path in `cfg/offline.py`.

```python
DISK_LOOP_PERIOD = 40.0
DISK_LOOP_REFERENCE_TIME = 20.0
DISK_LOOP_EPOCH = 0.0
DISK_LOOP_TEXTURE_B = 'cache/disk_native34_b.npy'
```

Bake B with the same material settings and a new seed, then render the preview.

```powershell
python tools/prebake_native34.py --seed 2718281 --output cache/disk_native34_b.npy
python tools/preview_disk.py --loop-frames 120 --velocity 0 0 0 --width 1024 --jitter 8 --step 0.02 --output output_frames/disk_loop_demo
ffmpeg -framerate 30 -i output_frames/disk_loop_demo/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output_frames/disk_loop_demo.mp4
```

This yields a 4-second preview of a coordinate-time period of 40 M.
`--time` chooses the observer start time, not the emission epoch.
`--save-linear` saves the LAST frame's HDR array; exposure overrides support
single-frame previews only. `cfg/offline_example.py` lists the same configuration
fields for longer offline runs. Current full-size A and B caches each contain
1 GiB of float32 coefficients.

## Validation

`tests/test_disk_loop.py` runs the actual CUDA kernel and checks period closure
at positive/negative times, visible A/B changes, the temporal join, and RGB
mixing before transfer. A radial-ray test compares against an independent
analytic Schwarzschild null travel-time integral:

    Delta_t(r) = F(r_camera)-F(r)
    F(r) = r-2 ln(r)+1/(4r)+4 ln(2r-1)
    F'(r) = (1+1/(2r))³/(1-1/(2r))

With zero extinction, constant A emissivity, dark B and a flat color LUT, a
separate dense quadrature predicts the observed RGB. This catches using
observer time for the blend or subtracting the light delay twice.

References: [isotropic Schwarzschild metric](https://www.einsteintoolkit.org/thornguide/EinsteinInitialData/IDAnalyticBH/documentation.html);
[Valve's staggered short-flow layers](https://cdn.akamai.steamstatic.com/apps/valve/2010/siggraph2010_vlachos_waterflow.pdf).
The disk equations and retarded-time construction above are derived for this
repository's existing conventions; Valve's method supplies the turnover idea.
