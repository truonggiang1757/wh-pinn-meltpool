"""Re-run evaluate_all on saved runs (model.pt) without retraining.
usage: python reeval.py <runs-subdir-glob>   e.g.  python reeval.py "E2/*/*"
"""
import os, sys, glob, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whpinn as C
from evaluate_common import evaluate_all
from runner import _clean, RUNS

_cache = {}


def load_rec(rd):
    ck = torch.load(os.path.join(rd, 'model.pt'), map_location=C.DEVICE, weights_only=False)
    cfg = ck['cfg']
    key = (cfg['case'], cfg.get('data_dir'), cfg.get('L_mode'))
    if key not in _cache:
        _cache[key] = C.load_data(cfg)
    data = _cache[key]
    if cfg['backbone'] == 'modified_mlp':
        model = C.ModifiedMLP(fourier=cfg['fourier'], sigma=cfg['fourier_sigma'])
    else:
        model = C.FlowNet(cfg['hidden_dims'], cfg['fourier'], cfg['fourier_sigma'])
    model.load_state_dict(ck['model']); model.to(C.DEVICE).eval()
    dist = None
    if ck['dist'] is not None:
        dist = C.DistanceNet(); dist.load_state_dict(ck['dist']); dist.to(C.DEVICE)
        for p in dist.parameters():
            p.requires_grad = False
    arc = C.ArcForce(0.5, 0.2, 1.0, 0.0, data['cl_x_min'], data['cl_x_max'])
    arc.load_state_dict(ck['arc']); arc.to(C.DEVICE)
    log_nu = ck['log_nu'].to(C.DEVICE)
    log_k = ck['log_kappa'].to(C.DEVICE) if ck['log_kappa'] is not None else None
    return C.Recon(model, dist, arc, log_nu, log_k, cfg, data), data, cfg


def split_for(cfg, data):
    n = len(data['sparse_xt'])
    tr = np.arange(n); va = np.array([], int); hi = np.array([], int); xt = None
    if cfg['holdout']:
        tr, va, _, _ = C.holdout_split(data, 42)
    if cfg['occlude_frac'] > 0:
        keep, hi, xt = C.occlusion_split(data, cfg['occlude_frac'])
        tr = np.intersect1d(tr, keep)
    return dict(train_idx=tr, val_idx=va, hidden_idx=hi, x_thr=xt)


if __name__ == '__main__':
    for rd in sorted(glob.glob(os.path.join(RUNS, sys.argv[1]))):
        if not os.path.exists(os.path.join(rd, 'model.pt')):
            continue
        rec, data, cfg = load_rec(rd)
        fields = dict(np.load(os.path.join(rd, 'fields.npz')))
        ev = evaluate_all(rec, data, None, split_for(cfg, data), fields, cfg['case'])
        json.dump(_clean(ev), open(os.path.join(rd, 'eval.json'), 'w'), indent=1)
        print('re-evaluated', rd, flush=True)
