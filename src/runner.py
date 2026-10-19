"""
Queue runner.  A queue file is a JSON list of specs:
  {"name": "E2/wirefed/final/s1", "variant": "final", "over": {"case": "wirefed", "seed": 1}}
Each run -> runs/<name>/{props.json, eval.json, fields.npz, history.npz, model.pt, log.txt}
Finished runs (eval.json present) are skipped, so a queue can be re-launched safely.

usage:  python runner.py <queue.json> [--shard i/n] [--dynamic]
  --shard i/n  run every n-th spec starting at i (static split over n processes)
  --dynamic    workers claim runs on the fly (start several processes on the same queue)
"""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from whpinn import make_cfg, run
from evaluate_common import evaluate_all

ROOT = paths.ROOT
RUNS = paths.RUNS


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_spec(spec):
    rd = os.path.join(RUNS, spec['name'])
    if os.path.exists(os.path.join(rd, 'eval.json')):
        print(f"skip {spec['name']}", flush=True)
        return
    cfg = make_cfg(spec['variant'], **spec.get('over', {}))
    t0 = time.time()
    rec, data, props, split = run(cfg, rd, verbose=False)
    fields = dict(np.load(os.path.join(rd, 'fields.npz')))
    ev = evaluate_all(rec, data, props, split, fields, cfg['case'])
    props['eval_time_s'] = time.time() - t0 - props['total_time_s']
    json.dump(_clean(props), open(os.path.join(rd, 'props.json'), 'w'), indent=1)
    json.dump(_clean(ev), open(os.path.join(rd, 'eval.json'), 'w'), indent=1)
    p = ev['ptv']['filt']
    print(f"done {spec['name']}: mu={props['mu_mPas']:.1f} umax={props['u_max']:.3f} "
          f"pde={props['final_losses']['pde']:.4f} MAE={ev['speed_MAE']:.4f} "
          f"D={p.get('D_PTV', float('nan')):.3f} Mmed={p.get('M_PTV_median', float('nan')):.2f} "
          f"Epde={ev['E_PDE_strong_common']:.4g} t={props['train_time_s']:.0f}s", flush=True)


def main():
    q = json.load(open(sys.argv[1]))
    if '--shard' in sys.argv:
        i, n = map(int, sys.argv[sys.argv.index('--shard') + 1].split('/'))
        q = q[i::n]
    dynamic = '--dynamic' in sys.argv    # workers claim runs via an atomic mkdir
    for spec in q:
        if dynamic:
            rd = os.path.join(RUNS, spec['name'])
            if os.path.exists(os.path.join(rd, 'eval.json')):
                continue
            try:
                os.makedirs(os.path.join(paths.CLAIMS, spec['name']))
            except FileExistsError:
                continue
        try:
            run_spec(spec)
        except Exception:
            print(f"FAILED {spec['name']}", flush=True)
            traceback.print_exc()
            os.makedirs(os.path.join(RUNS, spec['name']), exist_ok=True)
            open(os.path.join(RUNS, spec['name'], 'error.txt'), 'w').write(traceback.format_exc())


if __name__ == '__main__':
    main()
