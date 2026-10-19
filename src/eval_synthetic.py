"""E7 (synthetic recovery, Sec. 3.4.3): score the inversions of synthetic data against the known truth
-> results/synthetic_recovery.csv"""
import os, glob, json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ROOT = paths.ROOT
rows = []
for pj in sorted(glob.glob(os.path.join(paths.RUNS, 'E7', '*', '*', '*', 'props.json'))):
    rd = os.path.dirname(pj)
    p = json.load(open(pj))
    cfg = p['cfg']
    # dataset metadata (case, viscosity factor, noise level): shipped in data/synthetic/meta/<tag>/
    tag, sub = cfg['data_dir'].replace('\\', '/').rstrip('/').split('/')[-2:]
    meta = json.load(open(os.path.join(paths.SYNTH, 'meta', tag, sub, 'meta.json')))
    case, f = meta['case'], meta['factor']
    t = np.load(os.path.join(paths.SYNTH, f'truth_{case}', f'truth_nu{f:g}x.npz'))
    F = np.load(os.path.join(rd, 'fields.npz'))
    mask = t['mask'].astype(bool); ut, vt = t['u_ms'].astype(float), t['v_ms'].astype(float)
    rws, cls = np.where(mask)
    r0 = max(0, rws.min() - 10); c0 = max(0, cls.min() - 10)
    up = np.zeros_like(ut); vp = np.zeros_like(vt)
    h, w = F['u_ms'].shape
    up[r0:r0 + h, c0:c0 + w] = F['u_ms']; vp[r0:r0 + h, c0:c0 + w] = F['v_ms']
    m = mask
    rmse = float(np.sqrt(np.mean((up - ut)[m] ** 2 + (vp - vt)[m] ** 2)))
    trms = float(np.sqrt(np.mean(ut[m] ** 2 + vt[m] ** 2)))
    tmax = float(np.sqrt(ut ** 2 + vt ** 2)[m].max())
    nz = m & (np.sqrt(ut ** 2 + vt ** 2) > 0.02 * tmax)
    cos = (up * ut + vp * vt)[nz] / (np.hypot(up, vp)[nz] * np.hypot(ut, vt)[nz] + 1e-12)
    rows.append(dict(run=os.path.relpath(rd, ROOT).replace('\\', '/'), case=case, noise=meta['noise_rms'], factor=f,
                     seed=cfg['seed'], nu_true=float(t['nu_phys_true']), nu_rec=p['nu_phys'],
                     nu_ratio=p['nu_phys'] / float(t['nu_phys_true']),
                     umax_true=tmax, umax_rec=p['u_max'], umax_ratio=p['u_max'] / tmax,
                     U_char=p['U'], field_rmse=rmse, rmse_over_truth_rms=rmse / trms,
                     rmse_over_noise=rmse / meta['noise_rms'], median_cos=float(np.median(cos)),
                     mean_cos=float(np.mean(cos))))
df = pd.DataFrame(rows)
os.makedirs(paths.RESULTS, exist_ok=True)
df.to_csv(os.path.join(paths.RESULTS, 'synthetic_recovery.csv'), index=False)
print(df.to_string())
