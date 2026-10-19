#!/usr/bin/env python3
"""
Forward solver used to generate the synthetic ground truth (Sec. 3.4.3).

Solves steady, incompressible, dimensionless 2D Navier-Stokes on the actual
(pixelated) melt-pool geometry with the PINN-learned parameters:

    u.grad(u) = -grad(p) + nu_hat * lap(u) + F_tilde(x, y),   div(u) = 0

F_tilde is the learned radially-outward gradient-of-Gaussian source (identical
functional form to the ArcForce source in whpinn.py); nu_hat is the learned
dimensionless effective viscosity (1/Re).

Method: pseudo-transient Chorin projection on a staggered MAC grid
(exact discrete projection via DCT Poisson, no checkerboard modes),
first-order upwind (donor-cell) advection, explicit diffusion, implicit
Brinkman volume penalization of the solid region (no-penetration exact at
solid faces, no-slip emerging at the interface). March to steady state.

numpy + scipy only — no commercial software.
"""
import numpy as np
from scipy.fft import dctn, idctn

PIXEL_SIZE_M = 2.056640625e-05
CROP_PAD = 10


def load_geometry(mask_path, downsample):
    """Replicate load_data cropping/normalization; downsample for the solve."""
    mask = np.load(mask_path)
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - CROP_PAD); r1 = min(mask.shape[0], rows.max() + CROP_PAD)
    c0 = max(0, cols.min() - CROP_PAD); c1 = min(mask.shape[1], cols.max() + CROP_PAD)
    mask_crop = mask[r0:r1, c0:c1]
    rows_m, cols_m = np.where(mask_crop)
    cx = (cols_m.min() + cols_m.max()) / 2.0
    cy = (rows_m.min() + rows_m.max()) / 2.0
    W_m = (cols_m.max() - cols_m.min()) * PIXEL_SIZE_M
    H_m = (rows_m.max() - rows_m.min()) * PIXEL_SIZE_M
    L = max(W_m, H_m)

    ds = downsample
    H, W = mask_crop.shape
    Hd, Wd = H // ds, W // ds
    blocks = mask_crop[:Hd * ds, :Wd * ds].reshape(Hd, ds, Wd, ds)
    fluid = blocks.mean(axis=(1, 3)) > 0.5          # cell fluid if majority fluid

    # cell-centre coordinates in tilde units (same normalization as the PINN)
    ix = (np.arange(Wd) + 0.5) * ds - 0.5
    iy = (np.arange(Hd) + 0.5) * ds - 0.5
    xt = (ix - cx) * PIXEL_SIZE_M / L
    yt = (iy - cy) * PIXEL_SIZE_M / L
    h = ds * PIXEL_SIZE_M / L                       # grid spacing, tilde units
    return dict(mask_crop=mask_crop, fluid=fluid, xt=xt, yt=yt, h=h, L=L,
                cx=cx, cy=cy, crop=(r0, r1, c0, c1), ds=ds)


def arc_force(Xt, Yt, A_t, sigma_t, xc_t, yc_t=0.0):
    """Identical to ArcForce.forward: gradient-of-Gaussian, radially outward."""
    dx = Xt - xc_t
    dy = Yt - yc_t
    env = A_t * np.exp(-(dx**2 + dy**2) / (2.0 * sigma_t**2))
    return env * dx / sigma_t**2, env * dy / sigma_t**2


def dct_poisson_factory(Hd, Wd, h):
    """Neumann-box Poisson solver via DCT-II for the compact 5-point Laplacian."""
    lam_x = 2.0 * (np.cos(np.pi * np.arange(Wd) / Wd) - 1.0) / h**2
    lam_y = 2.0 * (np.cos(np.pi * np.arange(Hd) / Hd) - 1.0) / h**2
    LAM = lam_y[:, None] + lam_x[None, :]
    LAM[0, 0] = 1.0

    def solve(rhs):
        rhat = dctn(rhs, type=2, norm='ortho')
        rhat[0, 0] = 0.0
        return idctn(rhat / LAM, type=2, norm='ortho')
    return solve


def solve_steady(geo, nu_hat, A_t, sigma_t, xc_t,
                 eta=1e-4, cfl=0.4, max_steps=600_000,
                 tol=1e-5, check_every=2000, verbose=True):
    """
    MAC-grid pseudo-transient march to steady state.
      u: (Hd, Wd+1) at vertical faces;  v: (Hd+1, Wd) at horizontal faces.
    Returns cell-centred u, v, p (dimensionless) and diagnostics.
    """
    fluid = geo['fluid']; h = geo['h']
    xt, yt = geo['xt'], geo['yt']
    Hd, Wd = fluid.shape

    # --- face open/solid masks: a face is open iff both adjacent cells are fluid
    fl = fluid.astype(bool)
    u_open = np.zeros((Hd, Wd + 1), bool)
    u_open[:, 1:-1] = fl[:, 1:] & fl[:, :-1]
    v_open = np.zeros((Hd + 1, Wd), bool)
    v_open[1:-1, :] = fl[1:, :] & fl[:-1, :]

    # --- face-centre coordinates for the force
    xf = np.concatenate([[xt[0] - h / 2], (xt[:-1] + xt[1:]) / 2, [xt[-1] + h / 2]])
    Xu, Yu = np.meshgrid(xf, yt)                 # u-face centres (Hd, Wd+1)
    yf = np.concatenate([[yt[0] - h / 2], (yt[:-1] + yt[1:]) / 2, [yt[-1] + h / 2]])
    Xv, Yv = np.meshgrid(xt, yf)                 # v-face centres (Hd+1, Wd)
    Fu, _ = arc_force(Xu, Yu, A_t, sigma_t, xc_t)
    _, Fv = arc_force(Xv, Yv, A_t, sigma_t, xc_t)
    Fu *= u_open; Fv *= v_open

    u = np.zeros((Hd, Wd + 1)); v = np.zeros((Hd + 1, Wd))
    psolve = dct_poisson_factory(Hd, Wd, h)

    def lap(q):
        out = (np.roll(q, 1, 0) + np.roll(q, -1, 0) +
               np.roll(q, 1, 1) + np.roll(q, -1, 1) - 4 * q) / h**2
        return out

    hist = []
    u_prev = v_prev = None
    dt_diff = cfl * h**2 / (4 * nu_hat)
    for step in range(1, max_steps + 1):
        umax = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
        dt = min(cfl * h / umax, dt_diff)

        # ---- advection (donor cell) for u faces
        v_at_u = np.zeros_like(u)
        v_at_u[:, 1:-1] = 0.25 * (v[:-1, 1:] + v[:-1, :-1] + v[1:, 1:] + v[1:, :-1])
        dux_m = (u - np.roll(u, 1, 1)) / h; dux_p = (np.roll(u, -1, 1) - u) / h
        duy_m = (u - np.roll(u, 1, 0)) / h; duy_p = (np.roll(u, -1, 0) - u) / h
        adv_u = u * np.where(u > 0, dux_m, dux_p) + v_at_u * np.where(v_at_u > 0, duy_m, duy_p)

        # ---- advection for v faces
        u_at_v = np.zeros_like(v)
        u_at_v[1:-1, :] = 0.25 * (u[1:, :-1] + u[1:, 1:] + u[:-1, :-1] + u[:-1, 1:])
        dvx_m = (v - np.roll(v, 1, 1)) / h; dvx_p = (np.roll(v, -1, 1) - v) / h
        dvy_m = (v - np.roll(v, 1, 0)) / h; dvy_p = (np.roll(v, -1, 0) - v) / h
        adv_v = u_at_v * np.where(u_at_v > 0, dvx_m, dvx_p) + v * np.where(v > 0, dvy_m, dvy_p)

        ru = -adv_u + nu_hat * lap(u) + Fu
        rv = -adv_v + nu_hat * lap(v) + Fv
        us = u + dt * ru
        vs = v + dt * rv

        # ---- implicit Brinkman penalization on closed faces
        us /= (1.0 + (dt / eta) * (~u_open))
        vs /= (1.0 + (dt / eta) * (~v_open))

        # ---- projection (exact on MAC): div at cells, compact gradients at faces.
        # NOTE: solid faces are NOT re-clamped after projection — the implicit
        # penalization at the next step damps them; clamping here would undo
        # the projection and lock the solve at a spurious fixed point.
        div = (us[:, 1:] - us[:, :-1] + vs[1:, :] - vs[:-1, :]) / h
        phi = psolve(div / dt)
        u = us.copy(); v = vs.copy()
        u[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / h
        v[1:-1, :] -= dt * (phi[1:, :] - phi[:-1, :]) / h

        if step % check_every == 0:
            uc = 0.5 * (u[:, 1:] + u[:, :-1])
            vc = 0.5 * (v[1:, :] + v[:-1, :])
            sp = np.sqrt(uc**2 + vc**2)
            div2 = np.abs((u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / h)[fl].max()
            if u_prev is not None:
                # steady-state test: field change per checkpoint interval
                rel = max(np.abs(u - u_prev).max(), np.abs(v - v_prev).max()) \
                      / max(sp[fl].max(), 1e-9)
            else:
                rel = np.inf
            u_prev, v_prev = u.copy(), v.copy()
            hist.append((step, float(sp[fl].max()), float(rel)))
            if verbose:
                print(f"  step {step:7d} dt={dt:.2e} umax={sp[fl].max():.4f} "
                      f"div={div2:.2e} rel_change={rel:.3e}")
            if rel < tol and step > 10 * check_every:
                if verbose:
                    print(f"  converged at step {step}")
                break

    uc = 0.5 * (u[:, 1:] + u[:, :-1])
    vc = 0.5 * (v[1:, :] + v[:-1, :])
    uc *= fl; vc *= fl
    p = phi.copy()
    return uc, vc, p, dict(steps=step, history=hist)
