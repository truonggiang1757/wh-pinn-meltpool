"""
Common evaluation for every reconstruction: one metric set, same points,
same frame, for all models.
"""
import numpy as np
import torch
import os
import paths
from whpinn import (PIXEL_SIZE_M, DEVICE, site_keys, strong_residual, weak_residual)

PTV_FILES = {'wirefed': os.path.join(paths.PTV, 'ptv_wirefed.npz'),
             'autogenous': os.path.join(paths.PTV, 'ptv_autogenous.npz')}


def ptv_metrics(pu, pv, mu, mv):
    pm = np.hypot(pu, pv); mm = np.hypot(mu, mv)
    ok = (pm > 1e-4) & (mm > 1e-6)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()))
    PU, PV, MU, MV, PM, MM = pu[ok], pv[ok], mu[ok], mv[ok], pm[ok], mm[ok]
    cos = np.clip((PU * MU + PV * MV) / (PM * MM), -1, 1)
    ratio = MM / PM
    return dict(n=int(ok.sum()), D_PTV=float(cos.mean()), M_PTV_mean=float(ratio.mean()),
                M_PTV_median=float(np.median(ratio)),
                ang_median_deg=float(np.degrees(np.median(np.arccos(cos)))),
                vec_mae=float(np.hypot(MU - PU, MV - PV).mean()))


def eval_ptv(fields, data, case, x_thr_tilde=None):
    """Sample the crop-grid fields at every calibrated PTV observation (same rule as
    PTV/pinn_vs_ptv.py).  If x_thr_tilde is given, also score only the PTV points
    that lie inside the occluded (arc-side) region x_tilde >= x_thr."""
    P = np.load(PTV_FILES[case], allow_pickle=True)
    r0, r1, c0, c1 = data['crop']
    col = np.round(P['x_rect_px'] - c0).astype(int); row = np.round(P['y_rect_px'] - r0).astype(int)
    H, W = fields['mask'].shape
    ins = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    ins[ins] &= fields['mask'][row[ins], col[ins]].astype(bool)
    mu = np.zeros(len(col)); mv = np.zeros(len(col))
    mu[ins] = fields['u_ms'][row[ins], col[ins]]; mv[ins] = fields['v_ms'][row[ins], col[ins]]
    out = {}
    for tag, (pu, pv) in [('filt', (P['vx_filt_ms'], P['vy_filt_ms'])), ('raw', (P['vx_raw_ms'], P['vy_raw_ms']))]:
        pu = np.where(ins, pu, 0); pv = np.where(ins, pv, 0)
        out[tag] = ptv_metrics(pu, pv, mu, mv)
        if x_thr_tilde is not None and np.isfinite(x_thr_tilde):
            xt = (col - data['cx_px']) * PIXEL_SIZE_M / data['L']
            reg = ins & (xt >= x_thr_tilde)
            out[tag + '_hidden'] = ptv_metrics(np.where(reg, pu, 0), np.where(reg, pv, 0),
                                               np.where(reg, mu, 0), np.where(reg, mv, 0))
    return out


def table2_mae(fields, data, pct=0.9):
    """Replicates the original 'reconstruction_errors': errors against the per-pixel
    maximum-speed RAFT map built with the 0.9 dynamic-range threshold (sic)."""
    mask = data['mask_crop']; au, av = data['all_u'], data['all_v']
    h, w = mask.shape
    ms = np.zeros((h, w), np.float32); mu_ = np.zeros_like(ms); mv_ = np.zeros_like(ms); cnt = np.zeros((h, w), int)
    for i in range(au.shape[0]):
        mag = np.sqrt(au[i] ** 2 + av[i] ** 2)
        nt = mask & (mag > 1e-6)
        if nt.sum() == 0:
            continue
        lo, hi = mag[nt].min(), mag[nt].max()
        valid = nt & (mag >= lo + pct * (hi - lo))
        cnt[valid] += 1
        b = valid & (mag > ms)
        ms[b] = mag[b]; mu_[b] = au[i][b]; mv_[b] = av[i][b]
    om = cnt > 0
    sp = np.sqrt(fields['u_ms'] ** 2 + fields['v_ms'] ** 2)
    return dict(speed_MAE=float(np.abs(sp[om] - ms[om]).mean()),
                speed_RMSE=float(np.sqrt(((sp[om] - ms[om]) ** 2).mean())),
                vec_RMSE=float(np.sqrt(((fields['u_ms'][om] - mu_[om]) ** 2 + (fields['v_ms'][om] - mv_[om]) ** 2).mean())),
                n_pix=int(om.sum()))


def obs_metrics(rec, data, idx, prefix):
    """Metrics against a set of (instantaneous) observations `idx`, all computed on the
    SAME points: instantaneous vector/speed RMSE, site-mean RMSE, and the within-site
    temporal scatter of the observations themselves (vector and speed), plus E_OF."""
    if len(idx) == 0:
        return {}
    xy = np.stack([data['sparse_xt'][idx], data['sparse_yt'][idx]], 1)
    pu, pv = rec.predict_surface_SI(xy)
    out = {}
    # three groupings:  'orig'  = published 3-px |y| sites, raw v (mirror twins have opposite v)
    #                   'fold'  = same sites, v multiplied by sign(y) so mirror twins agree
    #                   'signed'= 3-px sites keyed on signed y (no mirror pooling)
    sgn = np.sign(data['sparse_iy'][idx] - data['cy_px']); sgn[sgn == 0] = 1
    qx = np.round((data['sparse_ix'][idx] - data['cx_px']) / 3.0).astype(np.int64)
    qys = np.round((data['sparse_iy'][idx] - data['cy_px']) / 3.0).astype(np.int64)
    for tag, keys, fold in [('orig', site_keys(data)[idx], False),
                            ('fold', site_keys(data)[idx], True),
                            ('signed', qx * 1000003 + qys + 500000, False)]:
        out.update(_obs_metrics_grouped(pu, pv, data['sparse_u'][idx], data['sparse_v'][idx],
                                        keys, sgn if fold else None, f'{prefix}_{tag}'))
    return out


def _obs_metrics_grouped(pu, pv, ou, ov, keys, sgn, prefix):
    if sgn is not None:
        pv = pv * sgn; ov = ov * sgn
    ps, os_ = np.hypot(pu, pv), np.hypot(ou, ov)
    uq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    def smean(a):
        s = np.zeros(len(uq)); np.add.at(s, inv, a); return s / cnt
    m_ou, m_ov, m_os = smean(ou), smean(ov), smean(os_)
    m_pu, m_pv, m_ps = smean(pu), smean(pv), smean(ps)
    r = {}
    r['n_points'] = int(len(ou)); r['n_sites'] = int(len(uq))
    r['inst_vec_RMSE'] = float(np.sqrt(np.mean((pu - ou) ** 2 + (pv - ov) ** 2)))
    r['inst_speed_RMSE'] = float(np.sqrt(np.mean((ps - os_) ** 2)))
    r['inst_speed_MAE'] = float(np.mean(np.abs(ps - os_)))
    # within-site scatter of the observations (same points)
    r['scatter_vec_RMS'] = float(np.sqrt(np.mean((ou - m_ou[inv]) ** 2 + (ov - m_ov[inv]) ** 2)))
    r['scatter_speed_RMS'] = float(np.sqrt(np.mean((os_ - m_os[inv]) ** 2)))
    # prediction vs site-mean observation: point-weighted (identity-consistent) and per-site
    r['sitemean_vec_RMSE_ptw'] = float(np.sqrt(np.mean((pu - m_ou[inv]) ** 2 + (pv - m_ov[inv]) ** 2)))
    r['sitemean_speed_RMSE_ptw'] = float(np.sqrt(np.mean((ps - m_os[inv]) ** 2)))
    r['sitemean_vec_RMSE_site'] = float(np.sqrt(np.mean((m_pu - m_ou) ** 2 + (m_pv - m_ov) ** 2)))
    r['sitemean_speed_RMSE_site'] = float(np.sqrt(np.mean((m_ps - m_os) ** 2)))
    # decomposition check: inst^2 ~= sitemean_ptw^2 + scatter^2 (+ within-site pred variation)
    r['identity_check_vec'] = float(np.sqrt(r['sitemean_vec_RMSE_ptw'] ** 2 + r['scatter_vec_RMS'] ** 2))
    r['E_OF'] = float(np.sqrt(np.sum((pu - ou) ** 2 + (pv - ov) ** 2) / np.sum(ou ** 2 + ov ** 2)))
    r['E_OF_sitemean'] = float(np.sqrt(np.sum((m_pu - m_ou) ** 2 + (m_pv - m_ov) ** 2) / np.sum(m_ou ** 2 + m_ov ** 2)))
    # high-speed bias: sites in the top quartile of observed site-mean speed
    q = m_os >= np.quantile(m_os, 0.75)
    r['topQ_site_speed_bias'] = float(np.mean(m_ps[q] - m_os[q]))
    r['topQ_site_speed_ratio'] = float(np.mean(m_ps[q] / np.maximum(m_os[q], 1e-9)))
    return {f'{prefix}_{k}': v for k, v in r.items()}


def common_residual(rec, data, n=4000, L_ref=None):
    """Physics residual evaluated the SAME way for every model: strong-form steady
    depth-averaged momentum residual (each model's own nu, drag and source), at a fixed
    set of n interior collocation points (seed 0), converted to SI (m/s^2) and then
    normalised by U^2/L_ref with L_ref = full characteristic length.  Also the weak-form
    residual with the default R/N_mc at the same centres and a fixed MC generator."""
    L_ref = L_ref or data['L_full']
    rng = np.random.RandomState(0)
    ii = rng.choice(len(data['interior_coords']), min(n, len(data['interior_coords'])), replace=False)
    # coordinates are in each model's own normalisation (data['L'])
    xy = torch.tensor(data['interior_coords'][ii], dtype=torch.float32, device=DEVICE)
    U, L = data['U'], data['L']
    out = {}
    rs = []
    for s in range(0, len(xy), 1000):
        rx, ry = strong_residual(rec, xy[s:s + 1000])
        rs.append((rx ** 2 + ry ** 2).detach().cpu().numpy().ravel())
    rs = np.concatenate(rs)
    to_ref = (U ** 2 / L) / (U ** 2 / L_ref)      # hat-residual -> SI -> ref-normalised
    out['E_PDE_strong_common'] = float(np.mean(rs) * to_ref ** 2)
    out['E_PDE_strong_common_median'] = float(np.median(rs) * to_ref ** 2)
    out['E_PDE_strong_SI_rms_ms2'] = float(np.sqrt(np.mean(rs)) * U ** 2 / L)
    g = torch.Generator(device=DEVICE); g.manual_seed(0)
    rw = []
    for s in range(0, len(xy), 500):
        rx, ry = weak_residual(rec, xy[s:s + 500], radius=0.05 * data['L_full'] / L, gen=g)
        rw.append((rx ** 2 + ry ** 2).detach().cpu().numpy().ravel())
    out['E_PDE_weak_common'] = float(np.mean(np.concatenate(rw)))
    # boundary compliance: surface speed on the extracted pool contour
    cnt = data['bnd_cnt_px']
    bxy = np.stack([(cnt[:, 0] - data['cx_px']) * PIXEL_SIZE_M / L,
                    (cnt[:, 1] - data['cy_px']) * PIXEL_SIZE_M / L], 1)
    bu, bv = rec.predict_surface_SI(bxy)
    out['wall_speed_mean'] = float(np.mean(np.hypot(bu, bv)))
    out['wall_speed_over_U'] = out['wall_speed_mean'] / U
    return out


def evaluate_all(rec, data, props, split, fields, case):
    ev = {}
    ev.update(table2_mae(fields, data))
    ev.update(obs_metrics(rec, data, split['train_idx'], 'train'))
    if len(split['val_idx']):
        ev.update(obs_metrics(rec, data, split['val_idx'], 'val'))
    if len(split['hidden_idx']):
        ev.update(obs_metrics(rec, data, split['hidden_idx'], 'hidden'))
    ev['ptv'] = eval_ptv(fields, data, case, split.get('x_thr'))
    ev.update(common_residual(rec, data))
    return ev
