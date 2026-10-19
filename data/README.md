# Data

## `observations/<case>/`: processed optical-flow observations (training data)

| Case | Regime | Frames | Pixel size | Frame step |
|---|---|---|---|---|
| `wirefed` | wire-fed (WAAM, wire touching the pool) | 91 | 10.53/512 mm | 5 frames at 1500 fps |
| `autogenous` | autogenous control | 99 | same | same |

Each case folder holds two files:

| File | Content |
|---|---|
| `mask_symmetric.npy` (included) | Pool mask (800 × 1280, bool) in the rectified, symmetrized frame |
| `frames.npz` (**download**) | The symmetrized RAFT optical-flow frames |

`frames.npz` contains:

| Key | Content |
|---|---|
| `frames` | (n, 2, h, w) float32. (u, v) in pixels per frame step, inside the bounding box `box` = (r0, r1, c0, c1) |
| `full_shape` | (800, 1280) |
| `frame_files` | The original frame names |

Every non-zero value lies inside `box`. `load_frames` in `src/whpinn.py` rebuilds the full frames, bit-identical to the preprocessing output.

**Download.** Get `observations_wirefed.npz` and `observations_autogenous.npz` from the [releases page of this repository](https://github.com/truonggiang1757/wh-pinn-meltpool/releases) (release `v1.0`). Save them as:

```
data/observations/wirefed/frames.npz
data/observations/autogenous/frames.npz
```

and verify them against `SHA256SUMS.txt`. The code also accepts the original preprocessing layout, `flow_symmetric/0000.npy` and so on, next to the mask.

## `ptv/`: independent particle tracks (never used in training)

### `ptv_wirefed.npz`

The 21 tracks (286 velocity vectors) of [19] on the same wire-fed recording, recalibrated into the present rectified frame.

### `ptv_autogenous.npz`

17 tracks (1,353 vectors) of the autogenous pool. These were tracked for this work with a learned point tracker (CoTracker3), and every position was confirmed manually.

Both files contain:

| Key | Content |
|---|---|
| `pid` | Track id |
| `t_s` | Time in seconds |
| `x_rect_px`, `y_rect_px` | Position in the rectified frame |
| `x_mm`, `y_mm` | Position in mm |
| `vx_raw_ms`, `vy_raw_ms` | Raw velocity |
| `vx_filt_ms`, `vy_filt_ms` | Velocity after Savitzky-Golay filtering |
| `pix_mm`, `arc_px`, `fps` | Frame scale, arc position and frame rate |

### `of_at_ptv_<case>.npz`

For each PTV vector, the rectified (non-symmetrized) optical flow at the same pixel and instant (`u_ms`, `v_ms`). It also gives whether that pixel passes the per-frame top-20 % selection used for training (`selected`). These are used for Table 6 and Fig. 15.

## `synthetic/`: synthetic recovery study (Sec. 3.4.3)

| Path | Content |
|---|---|
| `truth_<case>/truth_nu<f>x.npz` | Steady ground-truth surface flow on the real pool geometry, at 0.5×, 1×, 2× and 4× the recovered effective viscosity. Made by `src/gen_synthetic.py` with `src/solver.py` |
| `meta/<dataset>/data_nu<f>x/meta.json` | Case, viscosity factor and noise level of each degraded dataset |
| `example_frame_...0045.npz` | The degraded frame shown in Fig. 16b |

The degraded datasets themselves (about 750 MB each) are regenerated with `src/gen_synthetic.py`; see the main README. They go to `data/synthetic_obs/`, or to `$WHPINN_SYNTH_OBS`.
