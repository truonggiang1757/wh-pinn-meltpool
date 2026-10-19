"""Figures of the paper made from the training runs: Figs. 5, 6, 9, 10, 12, 13, 14, 15, 16 and the graphical
abstract -> figures/FigNN.png.  Wire-fed full-data WH-PINN = runs/E2/wirefed/wf_hard_F, autogenous = runs/E2/autogenous/final
(seed 42); the runs needed are shipped in runs/.  Figs. 1-4, 7, 8, 11, 17 and 18 are photographs, diagrams or
composites and are not generated here.

usage: python src/make_figures.py [fig5_6 fig9 ...] [--dpi 600]"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whpinn as C
from reeval import load_rec

import paths
ROOT = paths.ROOT
RUNS = paths.RUNS
OUT = paths.FIGURES
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9,
                     'axes.labelsize': 9, 'legend.fontsize': 7.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
PX_MM = C.PIXEL_SIZE_MM
DF = pd.read_csv(os.path.join(ROOT, 'results', 'all_runs.csv'))
_geo = {}


def geo(case):
    if case not in _geo:
        _geo[case] = C.load_data(C.make_cfg('final', case=case))
    return _geo[case]


def fields(run):
    return dict(np.load(os.path.join(RUNS, run, 'fields.npz')))


def props(run):
    return json.load(open(os.path.join(RUNS, run, 'props.json')))


def speed_panel(ax, run, case, title, vmax=None, star=True):
    d = geo(case); F = fields(run); p = props(run)
    m = F['mask'].astype(bool)
    h, w = m.shape
    x = np.arange(w) * PX_MM; y = np.arange(h) * PX_MM
    sp = np.hypot(F['u_ms'], F['v_ms'])
    vmax = vmax or sp[m].max()
    im = ax.pcolormesh(x, y, np.where(m, sp, np.nan), cmap='jet', shading='auto', vmin=0, vmax=vmax, alpha=0.6)
    cnt = d['bnd_cnt_px']; cc = np.vstack([cnt, cnt[:1]])
    ax.plot(cc[:, 0] * PX_MM, cc[:, 1] * PX_MM, 'k-', lw=0.8)
    X, Y = np.meshgrid(x, y)
    ax.streamplot(X, Y, np.where(m, F['u_ms'], 0), np.where(m, F['v_ms'], 0), color='k',
                  linewidth=0.45, density=1.4, arrowsize=0.4)
    if star:
        xa = (d['cx_px'] + p['arc_x_tilde'] * p['L'] / C.PIXEL_SIZE_M) * PX_MM
        ax.plot(xa, d['cy_px'] * PX_MM, 'r*', ms=6, zorder=5)
    ax.set_aspect('equal'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0])
    ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)'); ax.set_title(title, fontsize=8.5)
    return im


def musd(exp, case, group, col='mu_mPas'):
    s = DF[(DF.exp == exp) & (DF.case == case) & (DF.group == group)][col]
    return s.mean(), s.std(), len(s)


# ---------------------------------------------------------------------------------- Fig. 9
def fig9():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for row, (case, run, lab) in enumerate([('wirefed', 'E2/wirefed/wf_hard_F/s42', '(a) Wire-fed (WAAM)'),
                                            ('autogenous', 'E2/autogenous/final/s42', '(b) Autogenous (control)')]):
        H = dict(np.load(os.path.join(RUNS, run, 'history.npz'))); p = props(run)
        ep = np.arange(1, len(H['total']) + 1)
        ax = axes[row, 0]
        for k, l, c in [('total', r'$\mathcal{L}_{\rm total}$', 'k'), ('data', r'$\mathcal{L}_{\rm data}$', 'C0'),
                        ('pde', r'$\mathcal{L}_{\rm PDE}$', 'C3'), ('symmetry', r'$\mathcal{L}_{\rm sym}$', 'C1'),
                        ('p_ref', r'$\mathcal{L}_{\rm ref}$', 'C4'), ('arc_prior', r'$\mathcal{L}_{\rm prior}$', 'C2')]:
            ax.semilogy(ep, np.maximum(H[k], 1e-12), color=c, lw=0.8, label=l)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title(f'{lab}: training losses')
        ax.legend(ncol=3, loc='lower left', frameon=False)
        ax = axes[row, 1]
        L, U, RHO = p['L'], p['U'], C.RHO
        ax.plot(ep, H['nu_phys'] * 1e5, 'C0', lw=1.2, label=r'$\nu$ (closure, $\times10^{-5}$ m$^2$/s)')
        ax.plot(ep, np.array(H['arc_x']) * L * 1e3, 'C4', lw=1.2, label=r'$x_{\rm arc}$ (mm, from pool centre)')
        ax.plot(ep, np.array(H['arc_sigma']) * L * 1e3, 'C3--', lw=1.2, label=r'$\sigma$ (mm; prior 2.5)')
        ax.set_xlabel('Epoch'); ax.set_ylabel(r'$\nu$, $x_{\rm arc}$, $\sigma$')
        ax2 = ax.twinx()
        ax2.plot(ep, np.array(H['arc_amp']) * RHO * U ** 2 / L / 1e4, 'C2:', lw=1.4, label=r'$A$ ($\times10^4$ N/m$^3$, right)')
        ax2.set_ylabel(r'$A$ ($\times10^4$ N/m$^3$)')
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='center right', frameon=True, fontsize=7)
        ax.set_title(f'{lab}: closure and source parameters')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig09.png')); plt.close(fig)


# ---------------------------------------------------------------------------------- Fig. 10
def fig10():
    rows = [('baseline', '(a) Baseline: public code of [19]'), ('wf', '(b) + Weak form'),
            ('wf_fourier', '(c) + Fourier (soft boundary)'), ('wf_hard_noF', '(d) + Hard boundary (no Fourier)'),
            ('wf_hard_F', '(e) + Hard boundary + Fourier = WH-PINN')]
    vmax = max(np.hypot(fields(f'E2/wirefed/{g}/s42')['u_ms'], fields(f'E2/wirefed/{g}/s42')['v_ms']).max() for g, _ in rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))
    axes.ravel()[-1].axis('off')
    for ax, (g, t) in zip(axes.ravel(), rows):
        m, s, n = musd('E2', 'wirefed', g)
        im = speed_panel(ax, f'E2/wirefed/{g}/s42', 'wirefed', f'{t}\n$\\mu_{{\\rm eff}}$ = {m:.0f} ± {s:.0f} mPa·s (diagnostic; n = {n})', vmax)
    fig.subplots_adjust(right=0.9, hspace=0.75, wspace=0.35)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label('speed (m/s)')
    fig.savefig(os.path.join(OUT, 'Fig10.png')); plt.close(fig)


# ---------------------------------------------------------------------------------- Fig. 12
def wall_speed_profile(run):
    rec, d, cfg = load_rec(os.path.join(RUNS, run))
    cnt = d['bnd_cnt_px']
    xy = np.stack([(cnt[:, 0] - d['cx_px']) * C.PIXEL_SIZE_M / d['L'], (cnt[:, 1] - d['cy_px']) * C.PIXEL_SIZE_M / d['L']], 1)
    u, v = rec.predict_surface_SI(xy)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(cnt[:, 0]), np.diff(cnt[:, 1])))] * PX_MM
    return s, np.hypot(u, v) / d['U']


def fig12():
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1])
    vmax = max(np.hypot(fields(f'E2/wirefed/{g}/s42')['u_ms'], fields(f'E2/wirefed/{g}/s42')['v_ms']).max() for g in ['wf_fourier', 'wf_hard_F'])
    for ax, g, t in [(a1, 'wf_fourier', '(a) Soft boundary penalty (+ Fourier)'), (a2, 'wf_hard_F', '(b) DistanceNet hard boundary (+ Fourier)')]:
        w, ws, _ = musd('E2', 'wirefed', g, 'wall_speed_over_U')
        im = speed_panel(ax, f'E2/wirefed/{g}/s42', 'wirefed', f'{t}\nmean wall speed = {w:.2f} ± {ws:.2f} U', vmax, star=False)
    a3 = fig.add_subplot(gs[1, 0])
    for g, c, l in [('wf_fourier', 'C0', 'soft boundary'), ('wf_hard_F', 'C1', 'hard boundary')]:
        s, r = wall_speed_profile(f'E2/wirefed/{g}/s42')
        a3.plot(s, r, color=c, lw=0.9, label=l)
    a3.set_xlabel('arc length along extracted contour (mm)'); a3.set_ylabel('speed on contour / U')
    a3.set_title('(c) Reconstructed speed on the pool boundary (seed 42)'); a3.legend(frameon=False)
    a4 = fig.add_subplot(gs[1, 1])
    for g, c, l in [('wf_fourier', 'C0', 'soft boundary'), ('wf_hard_F', 'C1', 'hard boundary')]:
        Hs = np.array([np.load(f)['pde'] for f in sorted(glob.glob(os.path.join(RUNS, 'E2', 'wirefed', g, 's*', 'history.npz')))])
        k = 20; Hs = np.array([np.convolve(h, np.ones(k) / k, mode='valid') for h in Hs])
        ep = np.arange(Hs.shape[1]) + k
        a4.semilogy(ep, np.median(Hs, 0), color=c, lw=1.0, label=f'{l} (median of {len(Hs)} seeds)')
        a4.fill_between(ep, Hs.min(0), Hs.max(0), color=c, alpha=0.2)
    a4.set_xlabel('Epoch'); a4.set_ylabel('weak-form training residual')
    a4.set_title('(d) Training residual (20-epoch running mean; band = seed range)'); a4.legend(frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    pb = a2.get_position(); cax = fig.add_axes([0.92, pb.y0, 0.012, pb.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label('speed (m/s)')
    fig.savefig(os.path.join(OUT, 'Fig12.png')); plt.close(fig)


# ---------------------------------------------------------------------------------- Fig. 14
def fig14():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for row, (case, lab) in enumerate([('wirefed', 'wire-fed (WAAM)'), ('autogenous', 'autogenous (control)')]):
        run = f'E1/{case}/final_holdout/s42'
        rec, d, cfg = load_rec(os.path.join(RUNS, run))
        tr, va, _, ns = C.holdout_split(d, 42)
        ax = axes[row, 0]
        ax.scatter(d['interior_ix'] * PX_MM, d['interior_iy'] * PX_MM, s=0.2, c='#e8e2d8', rasterized=True)
        ax.scatter(d['sparse_ix'][tr] * PX_MM, d['sparse_iy'][tr] * PX_MM, s=0.5, c='#8a94a6', rasterized=True, label='training sites')
        ax.scatter(d['sparse_ix'][va] * PX_MM, d['sparse_iy'][va] * PX_MM, s=0.8, c='#d4430f', rasterized=True, label='held-out sites')
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
        ax.set_title(f'{lab}: site-level 80/20 split')
        ax.legend(markerscale=8, frameon=True, loc='lower right', fontsize=7)
        # per-site comparison, transverse velocity folded
        keys = C.site_keys(d)[va]
        uq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        xy = np.stack([d['sparse_xt'][va], d['sparse_yt'][va]], 1)
        pu, pv = rec.predict_surface_SI(xy)
        os_ = np.hypot(d['sparse_u'][va], d['sparse_v'][va]); ps = np.hypot(pu, pv)
        mo = np.bincount(inv, os_) / cnt; mp = np.bincount(inv, ps) / cnt
        so = np.sqrt(np.maximum(np.bincount(inv, os_ ** 2) / cnt - mo ** 2, 0))
        e = json.load(open(os.path.join(RUNS, run, 'eval.json')))
        ax = axes[row, 1]
        ax.errorbar(mo, mp, xerr=so, fmt='o', ms=2.2, color='#2b6cb0', ecolor='#9bb7d6', elinewidth=0.5, alpha=0.8,
                    label='held-out site (bar = within-site speed scatter)')
        lim = max(mo.max(), mp.max()) * 1.08
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='perfect agreement')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.set_xlabel('observed site-mean speed (m/s)'); ax.set_ylabel('predicted speed (m/s)')
        ax.set_title(f"held-out sites (n = {len(uq)}, seed 42)\ninstantaneous speed RMSE {e['val_fold_inst_speed_RMSE']:.3f}; "
                     f"obs. speed scatter {e['val_fold_scatter_speed_RMS']:.3f} m/s", fontsize=8)
        ax.legend(frameon=True, fontsize=6.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig14.png')); plt.close(fig)


# ---------------------------------------------------------------------------------- Fig. 16
def fig16():
    S = pd.read_csv(os.path.join(paths.RESULTS, 'synthetic_recovery.csv'))
    S = S[S.noise.isin([0.0266, 0.0221])]
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    # top row: wire-fed geometry, nu1x, measured noise, seed 42
    d = geo('wirefed'); r0, r1, c0, c1 = d['crop']
    t = np.load(os.path.join(paths.SYNTH, 'truth_wirefed', 'truth_nu1x.npz'))
    ut, vt = t['u_ms'][r0:r1, c0:c1], t['v_ms'][r0:r1, c0:c1]
    F = fields('E7/syn_wirefed_noise0027/nu1x/s42')
    m = d['mask_crop']; h, w = m.shape; x = np.arange(w) * PX_MM; y = np.arange(h) * PX_MM; X, Y = np.meshgrid(x, y)
    vmax = max(np.hypot(ut, vt)[m].max(), np.hypot(F['u_ms'], F['v_ms'])[m].max())
    cnt = d['bnd_cnt_px']; cc = np.vstack([cnt, cnt[:1]])
    obsdir = os.path.join(paths.SYNTH_OBS, 'syn_wirefed_noise0027', 'data_nu1x')
    if os.path.isdir(obsdir):                       # generated with gen_synthetic.py
        fr = C.load_frames(obsdir)[45]
    else:                                           # the one frame shown in the figure is shipped
        z = np.load(os.path.join(paths.SYNTH, 'example_frame_syn_wirefed_noise0027_nu1x_0045.npz'))
        b = [int(v) for v in z['box']]; fr = np.zeros((2, *[int(v) for v in z['full_shape']]), np.float32)
        fr[:, b[0]:b[1], b[2]:b[3]] = z['frame']
    fr = fr[:, r0:r1, c0:c1] * C.VEL_FACTOR
    mag = np.hypot(fr[0], fr[1]); nt = m & (mag > 1e-6); lo, hi = mag[nt].min(), mag[nt].max()
    sel = nt & (mag >= lo + 0.8 * (hi - lo))
    for k, (U_, V_, ttl) in enumerate([(ut, vt, '(a) known ground truth\n(rotational forcing)'), (None, None, '(b) degraded observations\n(one frame, 0.027 m/s noise, top 20 %)'),
                                       (F['u_ms'], F['v_ms'], '(c) WH-PINN reconstruction')]):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(cc[:, 0] * PX_MM, cc[:, 1] * PX_MM, 'k-', lw=0.7)
        if U_ is not None:
            im = ax.pcolormesh(x, y, np.where(m, np.hypot(U_, V_), np.nan), cmap='turbo', shading='auto', vmin=0, vmax=vmax)
            ax.streamplot(X, Y, np.where(m, U_, 0), np.where(m, V_, 0), color='k', linewidth=0.4, density=1.2, arrowsize=0.4)
        else:
            iy, ix = np.where(sel); kk = slice(None, None, max(1, len(ix) // 400))
            ax.quiver(ix[kk] * PX_MM, iy[kk] * PX_MM, fr[0][iy, ix][kk], -fr[1][iy, ix][kk], color='darkred', scale=4, width=0.004)
        ax.set_aspect('equal'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=8)
    top_last = fig.axes[-1]
    mk = {'wirefed': ('o', 'C0', 'wire-fed geometry'), 'autogenous': ('s', 'C3', 'autogenous geometry')}
    a = fig.add_subplot(gs[1, 0])
    for case, (mkr, c, l) in mk.items():
        g = S[S.case == case].groupby('factor')
        a.loglog(g.nu_true.mean(), g.nu_rec.mean(), marker=mkr, color=c, lw=1, label=l)
        a.scatter(S[S.case == case].nu_true, S[S.case == case].nu_rec, marker=mkr, color=c, s=8, alpha=0.5)
    lo_, hi_ = 5e-7, 4e-5
    a.plot([lo_, hi_], [lo_, hi_], 'k--', lw=0.8, label='perfect recovery')
    a.set_xlabel(r'true $\nu$ (m$^2$/s)'); a.set_ylabel(r'recovered $\nu$ (m$^2$/s)'); a.set_title('(d) viscosity identifiability')
    a.legend(frameon=False, fontsize=6.5)
    for k, (col, yl, ttl) in enumerate([('median_cos', 'median vector cosine', '(e) flow-pattern fidelity'),
                                        ('umax_ratio', 'recovered / true peak speed', '(f) peak-speed bias')]):
        a = fig.add_subplot(gs[1, k + 1])
        for j, (case, (mkr, c, l)) in enumerate(mk.items()):
            g = S[S.case == case].groupby('factor')[col]
            xs = np.arange(4) + (j - 0.5) * 0.3
            a.bar(xs, g.mean(), 0.3, color=c, alpha=0.8, label=l, yerr=g.std().fillna(0), capsize=2)
        a.set_xticks(range(4)); a.set_xticklabels(['0.5×', '1×', '2×', '4×']); a.set_xlabel(r'true $\nu$ (× learned value)')
        a.set_ylabel(yl); a.set_title(ttl)
        if col == 'median_cos':
            a.set_ylim(0.99, 1.0)
        else:
            a.axhline(1, color='k', lw=0.8, ls='--'); a.set_ylim(0, 1.8)
        a.legend(frameon=False, fontsize=6.5, loc='lower left' if col == 'median_cos' else 'upper left')
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    pb = top_last.get_position(); cax = fig.add_axes([0.92, pb.y0, 0.012, pb.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label('speed (m/s)')
    fig.savefig(os.path.join(OUT, 'Fig16.png')); plt.close(fig)


def fig5_6():
    """Figs. 5 and 6: flow field, learned (curl-free) source term, auxiliary gauge pressure. Seed 42."""
    for case, run, name in [('wirefed', 'E2/wirefed/wf_hard_F/s42', 'Fig05'), ('autogenous', 'E2/autogenous/final/s42', 'Fig06')]:
        d = geo(case); F = fields(run); p = props(run)
        m = F['mask'].astype(bool); h, w = m.shape
        x = np.arange(w) * PX_MM; y = np.arange(h) * PX_MM
        X, Y = np.meshgrid(x, y)
        L, U = p['L'], p['U']
        Xt = (np.arange(w) - d['cx_px']) * C.PIXEL_SIZE_M / L
        Yt = (np.arange(h) - d['cy_px']) * C.PIXEL_SIZE_M / L
        XT, YT = np.meshgrid(Xt, Yt)
        sig = p['arc_sigma_mm'] * 1e-3 / L
        dx, dy = XT - p['arc_x_tilde'], YT
        env = p['A_tilde'] * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
        scale = C.RHO * U ** 2 / L
        Fx, Fy = env * dx / sig ** 2 * scale, env * dy / sig ** 2 * scale
        Fm = np.hypot(Fx, Fy)
        xa = (d['cx_px'] + p['arc_x_tilde'] * L / C.PIXEL_SIZE_M) * PX_MM; ya = d['cy_px'] * PX_MM
        cnt = d['bnd_cnt_px']; cc = np.vstack([cnt, cnt[:1]])
        sp = np.hypot(F['u_ms'], F['v_ms'])
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
        ax = axs[0]
        im = ax.pcolormesh(x, y, np.where(m, sp, np.nan), cmap='jet', shading='auto', vmin=0, vmax=sp[m].max(), alpha=0.6)
        ax.streamplot(X, Y, np.where(m, F['u_ms'], 0), np.where(m, F['v_ms'], 0), color='k', linewidth=0.45, density=1.4, arrowsize=0.4)
        ax.plot(xa, ya, 'r*', ms=6, zorder=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('m/s'); cb.outline.set_visible(False)
        ax.set_title('(a) Flow field')
        ax = axs[1]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('wyr', [(0, 'white'), (0.25, 'lightyellow'), (0.5, 'orangered'), (1, 'darkred')])
        im = ax.pcolormesh(x, y, np.where(m, Fm / 1e4, np.nan), cmap=cmap, shading='auto')
        sk = max(1, min(h, w) // 20)
        mm = m[::sk, ::sk]
        ax.quiver(X[::sk, ::sk][mm], Y[::sk, ::sk][mm], Fx[::sk, ::sk][mm], -Fy[::sk, ::sk][mm], color='white',
                  scale=Fm[m].max() * 12, width=0.004, headwidth=3.5, alpha=0.85)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r'$\times10^4$ N/m$^3$'); cb.outline.set_visible(False)
        ax.set_title('(b) Source term (curl-free)')
        ax = axs[2]
        im = ax.pcolormesh(x, y, np.where(m, F['p_Pa'] - F['p_Pa'][m].min(), np.nan), cmap='RdBu_r', shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Pa'); cb.outline.set_visible(False)
        ax.set_title('(c) Auxiliary gauge pressure')
        for ax in axs:
            ax.plot(cc[:, 0] * PX_MM, cc[:, 1] * PX_MM, 'k-', lw=0.8)
            ax.set_aspect('equal'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0])
            ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
        fig.tight_layout(w_pad=1.6)
        fig.savefig(os.path.join(OUT, f'{name}.png')); plt.close(fig)


def of_at_ptv(case, P):
    """Rectified (not symmetrized) optical flow at each PTV vector's pixel and frame, and whether that pixel passes
    the per-frame top-20 % selection used for training.  Shipped as data/ptv/of_at_ptv_<case>.npz; recomputed
    from data/observations/<case>/flow_corrected/*.npy if that folder (preprocessing output of [19]) exists."""
    shipped = os.path.join(paths.PTV, f'of_at_ptv_{case}.npz')
    corr = os.path.join(paths.OBS, case, 'flow_corrected')
    if not os.path.isdir(corr):
        z = np.load(shipped)
        return z['u_ms'], z['v_ms'], z['selected']
    files = sorted(glob.glob(os.path.join(corr, '*.npy')))
    mask_full = np.load(os.path.join(paths.OBS, case, 'mask_symmetric.npy'))
    mid = P['t_s'] * float(P['fps']); fidx = np.clip(np.floor(mid / 5).astype(int), 0, len(files) - 1)
    n = len(mid); fu = np.zeros(n); fv = np.zeros(n); sel = np.zeros(n, bool); cache = {}
    for k, i in enumerate(fidx):
        if i not in cache:
            f = np.load(files[i]) * C.VEL_FACTOR; mag = np.hypot(f[0], f[1]); nt = mask_full & (mag > 1e-6)
            lo, hi = mag[nt].min(), mag[nt].max(); cache[i] = (f, nt & (mag >= lo + 0.8 * (hi - lo)))
        f, s = cache[i]; h, w = s.shape
        ix = int(np.clip(round(P['x_rect_px'][k]), 0, w - 1)); iy = int(np.clip(round(P['y_rect_px'][k]), 0, h - 1))
        fu[k], fv[k] = f[0, iy, ix], f[1, iy, ix]; sel[k] = s[iy, ix]
    return fu, fv, sel


def _ptv_sample(case):
    """PTV vectors with: reconstruction (seed 42) and time-matched rectified RAFT flow at the same
    pixel; per-vector training-selection flag (top-20 % of that frame, same rule as training)."""
    from evaluate_common import PTV_FILES
    P = dict(np.load(PTV_FILES[case], allow_pickle=True))
    fu, fv, sel = of_at_ptv(case, P)
    n = len(fu)
    d = geo(case); r0, r1, c0, c1 = d['crop']
    col = np.round(P['x_rect_px'] - c0).astype(int); row = np.round(P['y_rect_px'] - r0).astype(int)
    run = {'wirefed': 'E2/wirefed/wf_hard_F/s42', 'autogenous': 'E2/autogenous/final/s42'}[case]
    F = fields(run); m = F['mask'].astype(bool)
    ins = (row >= 0) & (row < m.shape[0]) & (col >= 0) & (col < m.shape[1])
    ins[ins] &= m[row[ins], col[ins]]
    mu = np.zeros(n); mv = np.zeros(n)
    mu[ins] = F['u_ms'][row[ins], col[ins]]; mv[ins] = F['v_ms'][row[ins], col[ins]]
    pu, pv = P['vx_filt_ms'], P['vy_filt_ms']

    def cosv(a, b, c, e):
        ok = (np.hypot(a, b) > 1e-4) & (np.hypot(c, e) > 1e-6)
        return np.clip((a * c + b * e)[ok] / (np.hypot(a, b) * np.hypot(c, e))[ok], -1, 1)
    return dict(P=P, F=F, d=d, col=col, row=row, sel=sel, run=run,
                cos_of=cosv(pu, pv, fu, fv), cos_rec=cosv(pu, pv, mu, mv))


def fig15():
    S = {c: _ptv_sample(c) for c in ['wirefed', 'autogenous']}
    D = {c: musd('E2', c, {'wirefed': 'wf_hard_F', 'autogenous': 'final'}[c], 'ptv.filt.D_PTV') for c in S}
    ceil = {'wirefed': '0.97', 'autogenous': '0.95 / 0.75'}
    lab = {'wirefed': 'wire-fed (0 mm)', 'autogenous': 'autogenous'}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax, c, t in [(axs[0, 0], 'wirefed', '(a)'), (axs[0, 1], 'autogenous', '(b)')]:
        s = S[c]; F = s['F']; m = F['mask'].astype(bool); h, w = m.shape
        x = np.arange(w) * PX_MM; y = np.arange(h) * PX_MM
        sp = np.hypot(F['u_ms'], F['v_ms'])
        im = ax.pcolormesh(x, y, np.where(m, sp, np.nan), cmap='turbo', shading='auto', vmin=0, vmax=sp[m].max())
        cnt = s['d']['bnd_cnt_px']; cc = np.vstack([cnt, cnt[:1]])
        ax.plot(cc[:, 0] * PX_MM, cc[:, 1] * PX_MM, 'k-', lw=0.7)
        pid = s['P']['pid']; cmap = plt.get_cmap('tab20')
        for j, p in enumerate(np.unique(pid)):
            idx = np.where(pid == p)[0]; idx = idx[np.argsort(s['P']['t_s'][idx])]
            xs, ys = s['col'][idx] * PX_MM, s['row'][idx] * PX_MM
            ax.plot(xs, ys, '-', color=cmap(j % 20), lw=0.8, alpha=0.9)
            ax.plot(xs[-1], ys[-1], 'o', color=cmap(j % 20), ms=2.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('reconstructed speed (m/s)')
        ntr = len(np.unique(pid)); nv = len(pid); dm, dsd, nn = D[c]
        ax.set_title(f'{t} {lab[c]}: {ntr} tracks, {nv:,} vectors\n'
                     f'$D_{{PTV}}$ = {dm:.2f} ± {dsd:.2f} ({nn} seeds); noise ceiling {ceil[c]}', fontsize=8)
        ax.set_aspect('equal'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0])
        ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
    # (c) autogenous training coverage
    ax = axs[1, 0]; s = S['autogenous']; d = s['d']; m = d['mask_crop']; h, w = m.shape
    freq = np.zeros((h, w)); np.add.at(freq, (d['sparse_iy'], d['sparse_ix']), 1); freq /= d['n_frames']
    x = np.arange(w) * PX_MM; y = np.arange(h) * PX_MM
    im = ax.pcolormesh(x, y, np.where(m, freq, np.nan), cmap='viridis', shading='auto')
    ax.scatter(s['col'] * PX_MM, s['row'] * PX_MM, s=0.6, c='red', label='PTV vectors')
    cnt = d['bnd_cnt_px']; cc = np.vstack([cnt, cnt[:1]]); ax.plot(cc[:, 0] * PX_MM, cc[:, 1] * PX_MM, 'k-', lw=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('training-selection frequency')
    ax.set_title(f'(c) autogenous: training-data coverage (top-20 % per frame)\n'
                 f'red: PTV vectors — {100 * s["sel"].mean():.1f} % pass the selection', fontsize=8)
    ax.set_aspect('equal'); ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[-1], y[0])
    ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
    # (d) per-vector direction agreement
    ax = axs[1, 1]
    for c, clr in [('wirefed', 'C0'), ('autogenous', 'C1')]:
        for key, ls, what in [('cos_of', '--', 'PTV vs optical flow'), ('cos_rec', '-', 'PTV vs reconstruction')]:
            v = np.sort(S[c][key])
            ax.plot(v, np.linspace(0, 1, len(v)), ls, color=clr, lw=1.1, label=f'{lab[c]}: {what}')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('cosine similarity per vector'); ax.set_ylabel('cumulative fraction')
    ax.set_title('(d) direction agreement per vector (seed 42)', fontsize=8)
    ax.legend(frameon=True, framealpha=1, edgecolor='none', fontsize=6.5, loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'Fig15.png')); plt.close(fig)
    for c in S:
        print(c, 'median cos OF %.3f, rec %.3f, mean rec %.3f, pass %.2f%%' % (np.median(S[c]['cos_of']), np.median(S[c]['cos_rec']),
                                                                             S[c]['cos_rec'].mean(), 100 * S[c]['sel'].mean()))


# ---------------------------------------------------------------------------------- Fig. 13 and graphical abstract
COL = {'baseline': '0.45', 'wf': 'C0', 'final': 'C3', 'wf_hard_noF': 'C1'}
LAB = {'baseline': 'Baseline (public code of [19])', 'wf': 'WF-PINN (weak form)',
       'final': 'WH-PINN', 'wf_hard_noF': 'WH-PINN without Fourier'}
SHORT = {'baseline': 'Baseline', 'wf': 'WF-PINN', 'final': 'WH-PINN', 'wf_hard_noF': 'WH-PINN\nno Fourier'}


def vals(exp, group, col):
    s = DF[(DF.exp == exp) & (DF.case == 'wirefed') & (DF.group == group)][col].dropna().values
    assert len(s) > 0, (exp, group, col)
    return s


def holdout():
    return {g: vals('E1', f'{g}_holdout', 'val_fold_E_OF') for g in ('baseline', 'wf', 'final')}


def occlusion():
    return {(g, f): vals('E4', f'{g}_occ{f}', 'hidden_fold_sitemean_speed_RMSE_site')
            for g in ('baseline', 'wf', 'final', 'wf_hard_noF') for f in (25, 50)}


def seed_plot(ax, x, v, color, w=0.08, ms=4.5):
    """mean +/- SD marker with the individual seeds as small dots."""
    rng = np.random.default_rng(0)
    ax.scatter(x + rng.uniform(-w, w, len(v)), v, s=7, color=color, alpha=0.55, lw=0, zorder=3)
    ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt='o', color=color, ms=ms, capsize=3, lw=1.1, zorder=4)


def panel_holdout(ax, H, annotate=True, fs=7):
    for i, g in enumerate(('baseline', 'wf', 'final')):
        v = H[g]
        seed_plot(ax, i, v, COL[g])
        if annotate:
            ax.text(i + 0.17, v.mean(), f'{v.mean():.3f}', va='center', fontsize=fs)
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[g] for g in ('baseline', 'wf', 'final')])
    ax.set_xlim(-0.5, 2.6); ax.grid(axis='y', alpha=0.3)


def panel_occ(ax, O, legend=True):
    groups = ('baseline', 'wf', 'final', 'wf_hard_noF')
    off = np.linspace(-0.27, 0.27, len(groups))
    for j, f in enumerate((25, 50)):
        for k, g in enumerate(groups):
            seed_plot(ax, j + off[k], O[(g, f)], COL[g], w=0.03, ms=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['25 % hidden', '50 % hidden'])
    ax.set_xlim(-0.5, 1.5); ax.grid(axis='y', alpha=0.3)
    if legend:
        h = [plt.Line2D([], [], marker='o', ls='', color=COL[g], label=SHORT[g].replace('\n', ' ')) for g in groups]
        ax.legend(handles=h, fontsize=6.5, frameon=True, framealpha=1, edgecolor='none', loc='upper left', ncol=2)


def fig13():
    runs = {'baseline': 'E2/wirefed/baseline/s42', 'wf': 'E2/wirefed/wf/s42'}
    vmax = max(np.hypot(fields(r)['u_ms'], fields(r)['v_ms']).max() for r in runs.values())
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.32,
                          left=0.08, right=0.9, top=0.95, bottom=0.08)
    for k, (g, t) in enumerate([('baseline', '(a) Baseline: public code of [19] (strong form)'),
                                ('wf', '(b) WF-PINN (weak form)')]):
        ax = fig.add_subplot(gs[0, k])
        im = speed_panel(ax, runs[g], 'wirefed', t, vmax)
    pos = ax.get_position()
    cax = fig.add_axes([0.915, pos.y0, 0.012, pos.height]); cb = fig.colorbar(im, cax=cax); cb.set_label('speed (m/s)')
    H, O = holdout(), occlusion()
    ax = fig.add_subplot(gs[1, 0]); panel_holdout(ax, H)
    ax.set_ylabel(r'$E_{\rm OF}$ at held-out sites'); ax.set_title('(c) Site-level hold-out (5 seeds)', fontsize=8.5)
    ax = fig.add_subplot(gs[1, 1]); panel_occ(ax, O)
    ax.set_ylabel('hidden-region site-mean\nspeed error (m/s)'); ax.set_title('(d) Arc-side occlusion (3 seeds)', fontsize=8.5)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    fig.savefig(os.path.join(OUT, 'Fig13.png')); plt.close(fig)
    for g, v in H.items():
        print(f'holdout {g:12s} E_OF {v.mean():.3f} ± {v.std(ddof=1):.3f} (n={len(v)})')
    for (g, f), v in O.items():
        print(f'occ{f} {g:12s} {v.mean():.3f} ± {v.std(ddof=1):.3f} (n={len(v)})')


def graphical_abstract():
    """Elsevier GA: >= 1328 x 531 px (w x h), readable at 13 x 5 cm."""
    H, O = holdout(), occlusion()
    with plt.rc_context({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        fig = plt.figure(figsize=(13 / 2.54 * 1.25, 5.2 / 2.54 * 1.25))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15], wspace=0.42, left=0.05, right=0.99, top=0.72, bottom=0.15)
        ax = fig.add_subplot(gs[0])
        speed_panel(ax, 'E2/wirefed/wf_hard_F/s42', 'wirefed', '')
        ax.set_title('Reconstructed surface flow\n(wire-fed pool, WH-PINN)', fontsize=7.5)
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1]); panel_holdout(ax, H, annotate=False)
        ax.set_title('Held-out error, $E_{\\rm OF}$\n(5 seeds)', fontsize=7.5)
        ax.set_xticklabels(['Base-\nline', 'Weak\nform', 'WH-\nPINN'])
        ax = fig.add_subplot(gs[2]); panel_occ(ax, O, legend=False)
        ax.set_title('Error with arc region hidden\n(m/s, 3 seeds)', fontsize=7.5)
        h = [plt.Line2D([], [], marker='o', ls='', ms=4, color=COL[g], label=s) for g, s in
             [('baseline', 'Baseline'), ('wf', 'Weak form'), ('final', 'WH-PINN'), ('wf_hard_noF', 'Hard wall only')]]
        ax.legend(handles=h, fontsize=5.8, ncol=2, frameon=False, loc='upper left', handletextpad=0.1, columnspacing=0.6)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        fig.suptitle('What a PINN can recover from molten-pool video: flow topology yes, effective viscosity no.\n'
                     'Wire-fed pool: the weak-form residual, not the full model, lowers the error on unseen regions.',
                     fontsize=7.5, y=0.985, linespacing=1.4)
        fig.savefig(os.path.join(OUT, 'Graphical_abstract.png')); plt.close(fig)


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--dpi' in args:
        i = args.index('--dpi'); plt.rcParams['savefig.dpi'] = int(args[i + 1]); del args[i:i + 2]
    for f in (args or ['fig5_6', 'fig9', 'fig10', 'fig12', 'fig13', 'fig14', 'fig15', 'fig16', 'graphical_abstract']):
        globals()[f](); print('wrote', f, flush=True)
