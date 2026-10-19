"""Rebuild the run queues of the paper from the configurations stored in runs/*/props.json.
Each spec is {name, variant, over}, where 'over' holds every setting that differs from make_cfg(variant), so
replaying a queue with runner.py reproduces the configuration of every reported run exactly.

  queues/paper_runs.json    E1-E5, E7  (hold-out, full data, alternatives, occlusion, sensitivity, synthetic)
  queues/timing_runs.json   E6, E6t    (training-time sessions; run alone on an idle GPU)

Experiment groups (runs/<exp>/<case>/<group>/s<seed>):
  E1  site-level hold-out (Tables 3-5, Figs. 13-14)      E2  full data (Tables 1, 2, 6; Figs. 5, 6, 9, 10, 12, 15)
  E3  ConFIG / modified MLP / LRA (Table 3)              E4  arc-side occlusion (Sec. 3.3.5, Fig. 13d)
  E5  sensitivity (Table 4)                              E7  synthetic recovery (Sec. 3.4.3, Fig. 16)
  E6, E6t  training time (Table 3, Sec. 3.3.5)"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from whpinn import make_cfg, VARIANTS

IGNORE = {'variant'}
specs = {'paper_runs.json': [], 'timing_runs.json': []}
for pj in sorted(glob.glob(os.path.join(paths.RUNS, '**', 'props.json'), recursive=True)):
    name = os.path.relpath(os.path.dirname(pj), paths.RUNS).replace('\\', '/')
    cfg = json.load(open(pj))['cfg']
    v = cfg['variant']
    if v not in VARIANTS:
        print('unknown variant', v, name); continue
    ref = make_cfg(v)
    over = {k: val for k, val in cfg.items() if k not in IGNORE and (k not in ref or ref[k] != val)}
    if over.get('data_dir'):                                   # synthetic datasets: relative to data/synthetic_obs
        tag, sub = over['data_dir'].replace('\\', '/').rstrip('/').split('/')[-2:]
        over['data_dir'] = f'data/synthetic_obs/{tag}/{sub}'
    key = 'timing_runs.json' if name.split('/')[0] in ('E6', 'E6t') else 'paper_runs.json'
    specs[key].append({'name': name, 'variant': v, 'over': over})
os.makedirs(os.path.join(paths.ROOT, 'queues'), exist_ok=True)
for fn, q in specs.items():
    json.dump(q, open(os.path.join(paths.ROOT, 'queues', fn), 'w'), indent=1)
    print(f'queues/{fn}: {len(q)} runs')
