#!/usr/bin/env python3
"""
Synthetic recovery study (Sec. 3.4.3, Fig. 16).  Truth = steady dimensionless Navier-Stokes on the REAL pool
geometry with a rotational, centerline-symmetric swirl-pair forcing (solver.py), at 0.5x, 1x, 2x and 4x the
effective viscosity recovered from the experiments; degraded through the real per-frame observation footprint,
additive noise at a given vector RMS, and the same mirror symmetrization as the preprocessing.

usage: python gen_synthetic.py <case> <noise_rms_vector_ms> <tag> [--truth-only <factor>]
  case : wirefed | autogenous
The four datasets of the paper:
  python src/gen_synthetic.py wirefed    0.0266 syn_wirefed_noise0027      # measured observation noise (paper)
  python src/gen_synthetic.py wirefed    0.1007 syn_wirefed_noise0101      # noise if mirrored data are pooled without folding v
  python src/gen_synthetic.py autogenous 0.0221 syn_autogenous_noise0022
  python src/gen_synthetic.py autogenous 0.0448 syn_autogenous_noise0045
Truth fields are cached in data/synthetic/truth_<case>/ (shipped); degraded frames (~750 MB per dataset) go to
data/synthetic_obs/<tag>/data_nu<f>x/ (or $WHPINN_SYNTH_OBS).  The noise RNG is seeded, so the frames are
reproduced exactly.
"""
import os, sys, json, glob, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import solver as S
from whpinn import load_frames

ROOT = paths.SYNTH            # truth fields (small, shipped)
DATA_ROOT = paths.SYNTH_OBS   # degraded frames (large, generated)
CASES = {
    # nu_hat = effective dimensionless viscosity and U = characteristic speed of the experimental reconstructions;
    # sigma_t, xc_t = learned source width/centre (dimensionless), reused as the swirl-pair width/centre
    'wirefed': dict(obs=os.path.join(paths.OBS, 'wirefed'), nu_hat=0.0019942051731050014,
                U=0.283284991979599, sigma_t=0.21437186002731323, xc_t=0.1555778980255127),
    'autogenous': dict(obs=os.path.join(paths.OBS, 'autogenous'), nu_hat=0.0016229472821578383,
                     U=0.0926380455493927, sigma_t=0.23153413832187653, xc_t=0.22126516699790955),
}
FACTORS = [0.5, 1.0, 2.0, 4.0]
TARGET_UMAX_HAT = 0.7
FPS, STRIDE = 1500.0, 5
PIX = S.PIXEL_SIZE_M
VEL_FACTOR = PIX / (STRIDE / FPS)
DS = 3
SWIRL_YC = 0.15


def make_swirl():
    def swirl_pair(Xt, Yt, A_t, sigma_t, xc_t, yc_t=0.0):
        ox = np.zeros_like(Xt); oy = np.zeros_like(Yt)
        for yc, s in ((SWIRL_YC, 1.0), (-SWIRL_YC, -1.0)):
            dx = Xt - xc_t; dy = Yt - yc
            env = A_t * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma_t ** 2))
            ox += s * (-env * dy / sigma_t ** 2); oy += s * (env * dx / sigma_t ** 2)
        return ox, oy
    return swirl_pair


S.arc_force = make_swirl()


def case_params(case):
    return dict(CASES[case])


def tune_amplitude(mask_path, nu_hat, sig, xc):
    geo = S.load_geometry(mask_path, downsample=6)
    A = 0.9
    for _ in range(6):
        u, v, _, _ = S.solve_steady(geo, nu_hat, A, sig, xc, max_steps=60000, check_every=5000, verbose=False)
        umax = float(np.sqrt(u ** 2 + v ** 2)[geo['fluid']].max())
        print(f"  [tune nu={nu_hat:.5f}] A={A:.4f} -> umax_hat={umax:.3f}", flush=True)
        if 0.5 <= umax <= 1.0:
            break
        A *= TARGET_UMAX_HAT / max(umax, 1e-6)
    return A


def upsample(field, geo, full_shape):
    ds = geo['ds']; r0, r1, c0, c1 = geo['crop']; Hc, Wc = geo['mask_crop'].shape
    big = np.repeat(np.repeat(field, ds, 0), ds, 1)
    crop = np.zeros((Hc, Wc)); h, w = min(Hc, big.shape[0]), min(Wc, big.shape[1])
    crop[:h, :w] = big[:h, :w]
    full = np.zeros(full_shape); full[r0:r1, c0:c1] = crop
    return full


def truth(case, f):
    c = case_params(case)
    td = os.path.join(ROOT, f'truth_{case}'); os.makedirs(td, exist_ok=True)
    fn = os.path.join(td, f'truth_nu{f:g}x.npz')
    if os.path.exists(fn):
        return fn
    mp = os.path.join(c['obs'], 'mask_symmetric.npy')
    mask = np.load(mp).astype(bool)
    nu_hat = f * c['nu_hat']
    A = tune_amplitude(mp, nu_hat, c['sigma_t'], c['xc_t'])
    geo = S.load_geometry(mp, downsample=DS)
    u, v, p, diag = S.solve_steady(geo, nu_hat, A, c['sigma_t'], c['xc_t'],
                                   max_steps=400000, check_every=5000, verbose=False)
    umax_hat = float(np.sqrt(u ** 2 + v ** 2)[geo['fluid']].max())
    uf = upsample(u, geo, mask.shape) * c['U'] * mask
    vf = upsample(v, geo, mask.shape) * c['U'] * mask
    np.savez_compressed(fn, u_ms=uf.astype(np.float32), v_ms=vf.astype(np.float32), mask=mask,
                        nu_hat_true=nu_hat, A_tilde=A, U_syn=c['U'], umax_hat=umax_hat,
                        nu_phys_true=nu_hat * c['U'] * geo['L'], L_m=geo['L'], steps=diag['steps'])
    print(f"  truth {case} nu{f:g}x: umax_hat={umax_hat:.3f} steps={diag['steps']}", flush=True)
    return fn


def degrade(case, f, noise_rms, tag, seed=42):
    c = case_params(case)
    t = np.load(truth(case, f))
    mask = t['mask'].astype(bool); full_shape = mask.shape
    uf, vf = t['u_ms'].astype(np.float64), t['v_ms'].astype(np.float64)
    rng = np.random.default_rng(seed)
    sig = noise_rms / np.sqrt(2.0)
    frames = load_frames(c['obs'])
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - 10); r1 = min(mask.shape[0], rows.max() + 10)
    c0 = max(0, cols.min() - 10); c1 = min(mask.shape[1], cols.max() + 10)
    cd = os.path.join(DATA_ROOT, tag, f'data_nu{f:g}x'); os.makedirs(os.path.join(cd, 'flow_symmetric'), exist_ok=True)
    for i, fr in enumerate(frames):
        fpm = mask & (np.sqrt(fr[0] ** 2 + fr[1] ** 2) * VEL_FACTOR > 1e-6)
        uo = uf + rng.normal(0, sig, full_shape); vo = vf + rng.normal(0, sig, full_shape)
        uc = uo[r0:r1, c0:c1]; vc = vo[r0:r1, c0:c1]; H = uc.shape[0]; top = np.arange(H // 2)
        uc[top, :] = uc[H - 1 - top, :]; vc[top, :] = -vc[H - 1 - top, :]
        uo[r0:r1, c0:c1] = uc; vo[r0:r1, c0:c1] = vc
        uo = np.where(fpm, uo, 0.0); vo = np.where(fpm, vo, 0.0)
        np.save(os.path.join(cd, 'flow_symmetric', f'{i:04d}.npy'),
                (np.stack([uo, vo]) / VEL_FACTOR).astype(np.float32))
    shutil.copy(os.path.join(c['obs'], 'mask_symmetric.npy'), os.path.join(cd, 'mask_symmetric.npy'))
    json.dump(dict(case=case, factor=f, noise_rms=noise_rms, n_frames=len(frames)),
              open(os.path.join(cd, 'meta.json'), 'w'))
    return cd


if __name__ == '__main__':
    case = sys.argv[1]
    if '--truth-only' in sys.argv:
        f = float(sys.argv[sys.argv.index('--truth-only') + 1]); truth(case, f); sys.exit()
    noise, tag = float(sys.argv[2]), sys.argv[3]
    for f in FACTORS:
        print('wrote', degrade(case, f, noise, tag), flush=True)
