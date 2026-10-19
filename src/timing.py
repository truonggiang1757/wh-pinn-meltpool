"""Training-cost comparison (Sec. 3.3.5, Table 3): median training time and peak GPU memory per model in each
timing session on the same laptop (RTX 4060 Laptop GPU).

  'E6t'          back-to-back session, 3 interleaved repeats per model (the values reported in the paper)
  'E6'           an earlier session (2 repeats)
  archive 'E6'   the first session (2 repeats; results/archive_rarg/all_runs.csv)

The small strong-form models are limited by CPU-side overhead, so their times depend on the CPU clock; only
times measured in one session should be compared.  Writes results/timing.json.
Run after: python src/aggregate.py"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

MODELS = ['final', 'wf', 'baseline', 'cao16', 'final_config', 'final_mmlp']
NAMES = {'final': 'WH-PINN', 'wf': 'WF-PINN', 'baseline': 'public baseline code of [19]',
         'cao16': 'formulation described in [19]', 'final_config': 'WH-PINN + ConFIG',
         'final_mmlp': 'WH-PINN + modified MLP'}
D = pd.read_csv(os.path.join(paths.RESULTS, 'all_runs.csv'))
A = pd.read_csv(os.path.join(paths.RESULTS, 'archive_rarg', 'all_runs.csv'))
sessions = {'session 1 (archive E6)': A[A.exp == 'E6'], 'session 2 (E6)': D[D.exp == 'E6'],
            'session 3 (E6t, paper)': D[D.exp == 'E6t']}

res = {}
for name, df in sessions.items():
    df = df[df.case == 'wirefed']
    res[name] = {}
    for g in MODELS:
        s = df[df.group == g]
        if len(s):
            res[name][g] = dict(n=int(len(s)), median_s=float(s.train_time_s.median()),
                                min_s=float(s.train_time_s.min()), max_s=float(s.train_time_s.max()),
                                peak_mem_GB=float(s.peak_mem_MB.median()) / 1024)
os.makedirs(paths.RESULTS, exist_ok=True)
json.dump(res, open(os.path.join(paths.RESULTS, 'timing.json'), 'w'), indent=1)


def cell(r):
    return f"{r['median_s']:.0f} s ({r['min_s']:.0f}-{r['max_s']:.0f}), n={r['n']}"


print(f"{'model':32s}" + ''.join(f'{k:>28s}' for k in res))
for g in MODELS:
    print(f'{NAMES[g]:32s}' + ''.join(f"{(cell(r[g]) if g in r else '-'):>28s}" for r in res.values()))
for k, r in res.items():
    if 'final' in r and 'baseline' in r:
        print(f"{k}: WH-PINN / baseline = {r['final']['median_s'] / r['baseline']['median_s']:.2f}")
