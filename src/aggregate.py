"""Collect every finished run into results/all_runs.csv and per-group mean/std tables."""
import os, glob, json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
ROOT = paths.ROOT


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{pre}{k}.'))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[pre + k] = v
    return out


rows = []
for ev in sorted(glob.glob(os.path.join(paths.RUNS, '**', 'eval.json'), recursive=True)):
    rd = os.path.dirname(ev)
    rel = os.path.relpath(rd, paths.RUNS).replace('\\', '/')
    parts = rel.split('/')
    p = json.load(open(os.path.join(rd, 'props.json')))
    e = json.load(open(ev))
    r = dict(run=rel, exp=parts[0], case=parts[1], group=parts[2], seed=parts[3])
    r.update({k: v for k, v in flat(p).items() if not k.startswith('cfg.')})
    r.update(flat(e))
    rows.append(r)
df = pd.DataFrame(rows).sort_values('run')
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
df.to_csv(os.path.join(ROOT, 'results', 'all_runs.csv'), index=False)

COLS = ['u_max', 'mu_mPas', 'Re', 'A_Nm3_as_reported', 'arc_sigma_mm', 'arc_x_mm', 'h_eff_mm',
        'final_losses.pde', 'pde_last50_mean', 'speed_MAE',
        'ptv.filt.D_PTV', 'ptv.filt.M_PTV_mean', 'ptv.filt.M_PTV_median', 'ptv.filt.ang_median_deg',
        'ptv.filt.vec_mae', 'ptv.raw.D_PTV',
        'val_fold_inst_vec_RMSE', 'val_fold_inst_speed_RMSE', 'val_fold_scatter_vec_RMS',
        'val_fold_scatter_speed_RMS', 'val_fold_sitemean_vec_RMSE_site', 'val_fold_sitemean_speed_RMSE_site',
        'val_orig_scatter_vec_RMS', 'val_fold_E_OF', 'val_fold_E_OF_sitemean',
        'val_fold_topQ_site_speed_ratio',
        'hidden_fold_sitemean_vec_RMSE_site', 'ptv.filt_hidden.D_PTV', 'ptv.filt_hidden.M_PTV_mean',
        'ptv.filt_hidden.M_PTV_median', 'ptv.filt_hidden.n',
        'E_PDE_strong_common', 'E_PDE_strong_common_median', 'E_PDE_weak_common', 'wall_speed_over_U',
        'train_time_s', 'peak_mem_MB', 'inference_time_s']
COLS = [c for c in COLS if c in df.columns]
g = df.groupby(['exp', 'case', 'group'])[COLS]
summ = g.agg(['mean', 'std', 'count'])
summ.to_csv(os.path.join(ROOT, 'results', 'summary_mean_std.csv'))


def fmt(m, s, n):
    if pd.isna(m):
        return ''
    if n > 1 and not pd.isna(s):
        return f'{m:.4g} ± {s:.2g}'
    return f'{m:.4g}'


lines = []
for (exp, case), sub in df.groupby(['exp', 'case']):
    lines.append(f'\n### {exp} / {case}\n')
    cols = [c for c in COLS if sub[c].notna().any()]
    lines.append('| group | n | ' + ' | '.join(cols) + ' |')
    lines.append('|' + '---|' * (len(cols) + 2))
    for grp, s2 in sub.groupby('group'):
        cells = [fmt(s2[c].mean(), s2[c].std(), s2[c].notna().sum()) for c in cols]
        lines.append(f'| {grp} | {len(s2)} | ' + ' | '.join(cells) + ' |')
open(os.path.join(ROOT, 'results', 'summary_tables.md'), 'w', encoding='utf8').write('\n'.join(lines))
print(f'{len(df)} runs aggregated')
