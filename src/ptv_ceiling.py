"""Monte-Carlo D_PTV / M_PTV 'noise ceiling': PTV = steady reconstruction + iid vector noise of
a given RMS, evaluated at the real PTV positions (same construction as the paper's 0.68)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, whpinn as C
from evaluate_common import PTV_FILES
out = {}
for case, rd in [('wirefed', 'E2/wirefed/wf_hard_F/s42'), ('autogenous', 'E2/autogenous/final/s42')]:
    d = C.load_data(C.make_cfg('final', case=case))
    F = np.load(os.path.join(paths.RUNS, rd, 'fields.npz'))
    P = np.load(PTV_FILES[case])
    r0, r1, c0, c1 = d['crop']
    col = np.round(P['x_rect_px'] - c0).astype(int); row = np.round(P['y_rect_px'] - r0).astype(int)
    ok = (row >= 0) & (row < F['mask'].shape[0]) & (col >= 0) & (col < F['mask'].shape[1])
    ok[ok] &= F['mask'][row[ok], col[ok]].astype(bool)
    mu, mv = F['u_ms'][row[ok], col[ok]], F['v_ms'][row[ok], col[ok]]
    res = {}
    # observation noise as measured (mirror twins folded) and as it appears if mirrored data are pooled without folding v
    for tag, rms in [('noise_pooled_without_folding', 0.1007 if case == 'wirefed' else 0.0442),
                     ('noise_measured', 0.0266 if case == 'wirefed' else 0.0221)]:
        rng = np.random.default_rng(0); D = []; M = []
        for _ in range(500):
            pu = mu + rng.normal(0, rms / np.sqrt(2), mu.shape); pv = mv + rng.normal(0, rms / np.sqrt(2), mv.shape)
            cos = (pu * mu + pv * mv) / (np.hypot(pu, pv) * np.hypot(mu, mv) + 1e-12)
            D.append(cos.mean()); M.append(np.median(np.hypot(mu, mv) / np.hypot(pu, pv)))
        res[tag] = dict(noise_rms=rms, D_ceiling=float(np.mean(D)), M_median_ceiling=float(np.mean(M)))
    out[case] = res
json.dump(out, open(os.path.join(paths.RESULTS, 'ptv_noise_ceiling.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
