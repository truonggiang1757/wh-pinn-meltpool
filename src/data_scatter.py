"""Observation scatter computed from the data alone (no model): the within-site repeatability of the optical-flow
observations (Sec. 3.4.1, Table 5) and the evaluation pitfall of pooling mirrored observations without folding v."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, whpinn as C
out = {}
for case in ['wirefed', 'autogenous']:
    d = C.load_data(C.make_cfg('final', case=case))
    tr, va, _, ns = C.holdout_split(d, 42)
    res = {'n_obs_total': int(len(d['sparse_u'])), 'n_frames': d['n_frames'], 'val_points': int(len(va)), 'n_sites': ns}
    for sub, idx in [('all', np.arange(len(d['sparse_u']))), ('val', va)]:
        u, v = d['sparse_u'][idx], d['sparse_v'][idx]
        sgn = np.sign(d['sparse_iy'][idx] - d['cy_px']); sgn[sgn == 0] = 1
        qx = np.round((d['sparse_ix'][idx] - d['cx_px']) / 3).astype(np.int64)
        qa = np.round(np.abs(d['sparse_iy'][idx] - d['cy_px']) / 3).astype(np.int64)
        qs = np.round((d['sparse_iy'][idx] - d['cy_px']) / 3).astype(np.int64)
        pix = (d['sparse_ix'][idx].astype(np.int64) * 100000 + d['sparse_iy'][idx])
        for tag, keys, vv in [('orig_absy_rawv', qx * 1000003 + qa, v), ('fold_absy_v_signy', qx * 1000003 + qa, v * sgn),
                              ('signed_y', qx * 1000003 + qs + 500000, v), ('per_pixel', pix, v)]:
            uq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            mu = np.bincount(inv, u) / cnt; mv = np.bincount(inv, vv) / cnt
            s = np.hypot(u, vv); ms = np.bincount(inv, s) / cnt
            rep = cnt[inv] > 1   # points whose group has >1 member
            res[f'{sub}_{tag}_vec_scatter'] = float(np.sqrt(np.mean((u - mu[inv]) ** 2 + (vv - mv[inv]) ** 2)))
            res[f'{sub}_{tag}_speed_scatter'] = float(np.sqrt(np.mean((s - ms[inv]) ** 2)))
            res[f'{sub}_{tag}_vec_scatter_repeated_only'] = float(np.sqrt(np.mean(((u - mu[inv]) ** 2 + (vv - mv[inv]) ** 2)[rep])))
            res[f'{sub}_{tag}_mean_group_size'] = float(cnt.mean())
    out[case] = res
json.dump(out, open(os.path.join(paths.RESULTS, 'data_scatter.json'), 'w'), indent=1)
for c, r in out.items():
    print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
