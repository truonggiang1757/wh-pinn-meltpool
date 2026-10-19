# WH-PINN: molten-pool surface flow from high-speed video

Code, data and results for the paper

> **What Can a Physics-Informed Neural Network Recover from Molten-Pool Video?**
> Truong Giang Bui Le, Yue Cao, Tuan Minh Vu, Van Anh Nguyen, Xuan Hai Le
> *Engineering Applications of Artificial Intelligence* (submitted)

**WH-PINN** is a weak-form, hard-constrained physics-informed neural network. It reconstructs the depth-averaged surface flow of a molten pool from sparse optical-flow observations. It combines three components:
- a weak-form Navier-Stokes residual;
- exact enforcement of the pool boundary through a pre-trained distance network;
- a Fourier input embedding.

The paper tests each design choice against independent data, using the same seeds, observations and metrics for every model. The data are two regimes of gas tungsten arc welding: a wire-fed pool (wire arc additive manufacturing, WAAM) and an autogenous pool.

The tests are:
- a leakage-free site-level hold-out;
- arc-side occlusion;
- independent particle tracking (PTV);
- synthetic recovery with a known ground truth;
- metallography.

## Contents

| Folder | Content |
|---|---|
| `src/` | All code: model and training (`whpinn.py`), queue runner, evaluation, aggregation, tables, figures, synthetic data |
| `queues/` | The exact run lists of the paper (`paper_runs.json`, 205 runs; `timing_runs.json`) and a small `example.json` |
| `data/observations/` | Pool masks. The processed optical-flow observations (`frames.npz`, ~150 MB per case) are a separate download, see [`data/README.md`](data/README.md) |
| `data/ptv/` | Independent particle tracks for both regimes, and the optical flow at each tracked vector |
| `data/synthetic/` | Ground-truth fields and metadata of the synthetic recovery study |
| `runs/` | `props.json` and `eval.json` of all 231 runs of the paper. Full outputs (fields, training history, model weights) are included for the runs that the figures use |
| `results/` | `all_runs.csv` (every metric of every run), the paper tables, key statistics, timing, synthetic-recovery scores |

## Installation

Tested with Python 3.13, PyTorch 2.11 (CUDA 12.8) and an RTX 4060 Laptop GPU. A CPU works for everything except retraining at full length.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build, see pytorch.org
pip install -r requirements.txt
```

Then download the two observation files from the [releases page](https://github.com/truonggiang1757/wh-pinn-meltpool/releases), as described in [`data/README.md`](data/README.md).

## Quick start

Train WH-PINN once on each pool, at full length with seed 42:

```bash
python src/runner.py queues/example.json
```

Each run takes a few minutes on the GPU and writes the following to `runs/example/<case>/final/s42/`:

| File | Content |
|---|---|
| `props.json` | Configuration, recovered parameters, timing |
| `eval.json` | All metrics against hold-out, PTV, residual and wall |
| `fields.npz` | Velocity and pressure on the pixel grid |
| `history.npz` | Training curves |
| `model.pt` | Model weights |

## Reproducing the paper

### Without retraining

This works from the shipped run outputs:

```bash
python src/aggregate.py        # runs/*/eval.json -> results/all_runs.csv, summary_tables.md
python src/eval_synthetic.py   # synthetic-recovery scores -> results/synthetic_recovery.csv
python src/timing.py           # training-time sessions -> results/timing.json
python src/data_scatter.py     # observation repeatability (Sec. 3.4.1, Table 5)
python src/ptv_ceiling.py      # noise ceilings of the PTV direction agreement (Table 6)
python src/make_tables.py      # Tables 1-6 -> results/paper_tables.md, key_statistics.json
python src/make_figures.py     # Figs. 5, 6, 9, 10, 12-16 and the graphical abstract -> figures/
```

### Retraining every model

```bash
# synthetic datasets (about 750 MB each; the degradation is seeded, so the frames are reproduced exactly)
python src/gen_synthetic.py wirefed    0.0266 syn_wirefed_noise0027
python src/gen_synthetic.py wirefed    0.1007 syn_wirefed_noise0101
python src/gen_synthetic.py autogenous 0.0221 syn_autogenous_noise0022
python src/gen_synthetic.py autogenous 0.0448 syn_autogenous_noise0045

# the 205 runs of the paper (roughly 15 h on one RTX 4060 laptop GPU)
WHPINN_RUNS=runs_new python src/runner.py queues/paper_runs.json
# training-time sessions: run alone on an otherwise idle machine
WHPINN_RUNS=runs_new python src/runner.py queues/timing_runs.json
```

Notes:
- The runner skips any run whose `eval.json` already exists. Setting `WHPINN_RUNS` to a new folder therefore avoids skipping the shipped runs.
- To use several processes on one queue, start each one with `--dynamic`.
- After retraining, point the analysis scripts at the new runs. Set `WHPINN_RUNS` for the analysis commands above as well.
- GPU kernels are not bitwise deterministic, so retrained metrics agree with the paper within the seed-to-seed spread, not to the last digit.

## Experiments and where they appear in the paper

Runs are named `runs/<experiment>/<case>/<group>/s<seed>`. The cases are `wirefed` (the wire-fed WAAM pool, wire height 0 mm) and `autogenous` (the autogenous pool). Synthetic datasets are named by geometry and noise level, for example `syn_wirefed_noise0027` (vector noise 0.027 m/s).

| Experiment | What | Seeds | Paper |
|---|---|---|---|
| E1 | Site-level hold-out: WH-PINN, WF-PINN, public baseline, formulation of [19] | 5 | Tables 3, 5; Figs. 13, 14 |
| E2 | Full data: ablation ladder, both regimes | 5 | Tables 1, 2, 6; Figs. 5, 6, 9, 10, 12, 15 |
| E3 | WH-PINN + ConFIG, + modified MLP, + learning-rate annealing | 3 | Table 3 |
| E4 | Arc-side 25 % / 50 % of observation sites hidden | 3 | Sec. 3.3.5, Fig. 13d |
| E5 | One-at-a-time sensitivity (Fourier scale, sub-domain radius, Monte Carlo points, loss weights) | 3 | Table 4 |
| E6, E6t | Training time; E6t = back-to-back session reported in the paper | 2–3 | Table 3, Sec. 3.3.5 |
| E7 | Synthetic recovery: 2 geometries × 2 noise levels × 4 viscosities | 2 | Sec. 3.4.3, Fig. 16 |

## Model variants

All variants are defined in `src/whpinn.py` (`VARIANTS`, `BASE_CFG`).

| Variant | Residual | Boundary | Fourier | Notes |
|---|---|---|---|---|
| `baseline` | strong | soft penalty | no | the public code of Cao et al. [19], re-implemented in this framework |
| `wf` | weak | soft | no | WF-PINN |
| `wf_fourier` | weak | soft | yes | |
| `wf_hard_noF` | weak | hard (DistanceNet) | no | |
| `wf_hard_F` = `final` | weak | hard | yes | **WH-PINN** |
| `cao16` | strong | soft | no | our implementation of the formulation *described* in [19]: fixed viscosity, linear drag, 3/2 surface-to-depth ratio, ConFIG |

Switches set through a queue entry's `over`:

| Switch | Effect |
|---|---|
| `holdout` | Site-level split |
| `occlude_frac` | Hide the arc-side fraction of observation sites |
| `balancing` | `fixed`, `config` or `lra` |
| `backbone` | `mlp` or `modified_mlp` |
| `fourier_sigma` | Fourier embedding scale |
| `mc_radius`, `mc_samples` | Weak-form sub-domain radius and Monte Carlo points |
| `lambda_*` | Loss weights |
| `disable_arc_force` | Turn the source term off |
| `rar` | Residual-based adaptive sampling. This was tested in a preliminary study and had no measurable effect (`results/rarg_comparison.csv`), so it is off for every run in the paper. |

## Notes

- **Timing.** The small strong-form models (the baseline and the formulation of [19]) are limited by CPU-side overhead. Their training time therefore depends on the CPU clock, and only times measured in one session should be compared (`results/timing.json`).
- **Preprocessing.** The video-to-observation pipeline (masking, RAFT optical flow, homography calibration, symmetrization) is steps 0–3 of the public code of [19], [github.com/YueCaoUKY/Molten-Pool-Flow-Detection-via-PINN](https://github.com/YueCaoUKY/Molten-Pool-Flow-Detection-via-PINN). The only change is in step 2: the pool contour is extracted with `cv2.CHAIN_APPROX_NONE` instead of `CHAIN_APPROX_SIMPLE`. The raw high-speed videos are available from the corresponding authors on reasonable request.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use the code or data.

## License

- Code: MIT ([`LICENSE`](LICENSE)).
- Data in `data/` and the observation files: CC BY 4.0.
