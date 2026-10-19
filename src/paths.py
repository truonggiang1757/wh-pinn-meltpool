"""Repository paths. Everything is relative to the repository root; each location can be
redirected with an environment variable (e.g. to keep the large synthetic data on another disk)."""
import os

ROOT = os.environ.get('WHPINN_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OBS = os.environ.get('WHPINN_OBS', os.path.join(DATA, 'observations'))          # processed optical-flow observations
PTV = os.path.join(DATA, 'ptv')                                                  # independent particle tracks
SYNTH = os.path.join(DATA, 'synthetic')                                          # synthetic truth fields + metadata
SYNTH_OBS = os.environ.get('WHPINN_SYNTH_OBS', os.path.join(DATA, 'synthetic_obs'))  # degraded synthetic frames (~750 MB each)
RUNS = os.environ.get('WHPINN_RUNS', os.path.join(ROOT, 'runs'))
CLAIMS = os.path.join(os.path.dirname(RUNS), 'claims')                          # used by runner.py --dynamic
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
