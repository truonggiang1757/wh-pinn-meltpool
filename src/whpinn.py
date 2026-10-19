"""
whpinn.py -- WH-PINN (variant 'final') and every formulation it is compared with, in one
parameterised module.  The numerics (data loading, Fourier map, DistanceNet, ArcForce source,
weak/strong residuals, loss terms, optimiser, schedule) follow the original single-purpose
training scripts; only switches were added:

  form        'weak' | 'strong'
  fourier     bool (sigma_B = fourier_sigma)
  hard_bc     bool (psi = D^2 * psi_raw)  else soft boundary band penalty
  rar         bool (EMA residual-based adaptive sampling; tested, inert, OFF in the paper)
  balancing   'fixed' | 'config' (ConFIG over data/pde/bnd) | 'lra' (Wang 2021 annealing)
  backbone    'mlp' | 'modified_mlp'
  cao_mode    bool  -> our implementation of the formulation DESCRIBED in Cao et al. [19]:
                       fixed nu = 7.1e-7 m^2/s, learnable linear drag kappa,
                       u_s = 1.5 u (half-Poiseuille), L = max(W,H)/2, ConFIG,
                       strong form, soft boundary, 800 epochs, 3000 colloc/epoch
  holdout     bool  -> site-level 80/20 split (sites pool a location, its mirror twin and
                       all temporal repeats)
  occlude_frac float -> hide the arc-side fraction of observation SITES
  seed        int

Every run writes <run_dir>/props.json, fields.npz, model.pt, history.npz.
"""
import os, glob, json, time, math
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.ndimage import distance_transform_edt
import paths

PIXEL_SIZE_MM = 10.53 / 512
PIXEL_SIZE_M = PIXEL_SIZE_MM * 1e-3
FPS, STRIDE = 1500, 5
DT = STRIDE / FPS
RHO = 7200.0
VEL_FACTOR = PIXEL_SIZE_M / DT
DEVICE = os.environ.get('WHPINN_DEVICE') or ('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

# processed observations per case: <dir>/mask_symmetric.npy and either <dir>/frames.npz
# (packed, as distributed) or <dir>/flow_symmetric/0000.npy ... (layout of the preprocessing code of [19])
CASE_DIRS = {'wirefed': 'wirefed', 'autogenous': 'autogenous'}   # wire-fed pool (WAAM, wire height 0 mm), autogenous pool


def load_frames(data_dir):
    """All symmetrized optical-flow frames of a case as float32 arrays (2, H, W) in px/frame-step.
    frames.npz stores only the bounding box that contains every non-zero value; the full frames
    are rebuilt with zeros elsewhere, i.e. bit-identical to the flow_symmetric/*.npy files."""
    packed = os.path.join(data_dir, 'frames.npz')
    if os.path.exists(packed):
        z = np.load(packed)
        r0, r1, c0, c1 = (int(v) for v in z['box']); H, W = (int(v) for v in z['full_shape'])
        out = []
        for crop in z['frames']:
            f = np.zeros((2, H, W), np.float32); f[:, r0:r1, c0:c1] = crop
            out.append(f)
        return out
    files = sorted(glob.glob(os.path.join(data_dir, 'flow_symmetric', '*.npy')))
    if not files:
        raise FileNotFoundError(f'no observations in {data_dir}: expected frames.npz or flow_symmetric/*.npy '
                                '(see data/README.md)')
    return [np.load(f) for f in files]

BASE_CFG = {
    'case': 'wirefed', 'data_dir': None,
    'crop_padding': 10,
    'boundary_band_width': 50, 'boundary_inner_fraction': 0.5, 'boundary_outer_fraction': 0.5,
    'n_obs_per_step': 2000, 'n_pde_per_step': 1500, 'n_boundary_points': 800,
    'lambda_data': 5.0, 'lambda_boundary': 2.0, 'lambda_pde': 1.0,
    'lambda_p_ref': 0.5, 'lambda_symmetry': 2.0, 'lambda_peak': 0.1,
    'disable_arc_force': False,
    'arc_x_frac_init': 0.633, 'arc_sigma_init_m': 2.5e-3, 'arc_amp_init': 1.0,
    'lambda_arc_pos_prior': 2.0, 'arc_x_frac_measured': 0.726, 'arc_x_frac_tolerance': 0.05,
    'lambda_arc_sigma_prior': 1.0, 'arc_sigma_prior_log_std': 0.3,
    'n_epochs': 1000, 'lr': 1e-3, 'lr_physics': 5e-3,
    'hidden_dims': [128, 256, 256, 128],
    'speed_range_percentile': 0.8,
    'mc_samples': 16, 'mc_radius': 0.05,
    # switches
    'form': 'weak', 'fourier': True, 'fourier_sigma': 0.5, 'hard_bc': True,
    'rar': False, 'rar_c': 0.1, 'rar_c_rel': None, 'rar_alpha': 0.9,   # adaptive sampling OFF (tested; inert)
    'balancing': 'fixed', 'backbone': 'mlp',
    'cao_mode': False, 'L_mode': 'full', 'surface_factor': 1.0,
    'fixed_nu_phys': None, 'drag': False,
    'holdout': False, 'occlude_frac': 0.0,
    'seed': 42, 'dist_steps': 2000,
}

# the ablation ladder of Table 2, WH-PINN, and the formulation described in [19]
VARIANTS = {
    'baseline':       dict(form='strong', fourier=False, hard_bc=False, rar=False),
    'wf':             dict(form='weak',   fourier=False, hard_bc=False, rar=False),
    'wf_fourier':     dict(form='weak',   fourier=True,  hard_bc=False, rar=False),
    'wf_hard_noF':    dict(form='weak',   fourier=False, hard_bc=True,  rar=False),
    'wf_hard_F':      dict(form='weak',   fourier=True,  hard_bc=True,  rar=False),
    # WH-PINN ('final' == 'wf_hard_F': weak form + hard boundary + Fourier features, uniform sampling)
    'final':          dict(form='weak',   fourier=True,  hard_bc=True,  rar=False),
    # with residual-based adaptive sampling (preliminary study; not used in the paper)
    'final_rar':      dict(form='weak',   fourier=True,  hard_bc=True,  rar=True),
    'wf_rar':         dict(form='weak',   fourier=False, hard_bc=False, rar=True),
    'cao16':          dict(form='strong', fourier=False, hard_bc=False, rar=False,
                           cao_mode=True, balancing='config', L_mode='half',
                           surface_factor=1.5, fixed_nu_phys=7.1e-7, drag=True,
                           n_epochs=800, n_pde_per_step=3000),
}


def make_cfg(variant='final', **over):
    c = dict(BASE_CFG)
    c.update(VARIANTS[variant])
    c.update(over)
    c['variant'] = variant
    return c


# ----------------------------------------------------------------------------- data
def load_data(cfg):
    data_dir = cfg['data_dir'] or os.path.join(paths.OBS, CASE_DIRS[cfg['case']])
    if not os.path.isabs(data_dir):                       # relative paths (synthetic datasets in the queues)
        rel = data_dir.replace('\\', '/')
        data_dir = (os.path.join(paths.SYNTH_OBS, rel[len('data/synthetic_obs/'):])
                    if rel.startswith('data/synthetic_obs/') else os.path.join(paths.ROOT, rel))
    mask = np.load(os.path.join(data_dir, 'mask_symmetric.npy'))
    rows, cols = np.where(mask)
    pad = cfg['crop_padding']
    r0 = max(0, rows.min() - pad); r1 = min(mask.shape[0], rows.max() + pad)
    c0 = max(0, cols.min() - pad); c1 = min(mask.shape[1], cols.max() + pad)
    mask_crop = mask[r0:r1, c0:c1]
    h_c, w_c = mask_crop.shape
    rows_m, cols_m = np.where(mask_crop)
    cx_px = (cols_m.min() + cols_m.max()) / 2.0
    cy_px = (rows_m.min() + rows_m.max()) / 2.0
    W_m = (cols_m.max() - cols_m.min()) * PIXEL_SIZE_M
    H_m = (rows_m.max() - rows_m.min()) * PIXEL_SIZE_M
    L_full = max(W_m, H_m)
    L = L_full / 2.0 if cfg.get('L_mode') == 'half' else L_full

    frames = load_frames(data_dir)
    n_frames = len(frames)
    ix_l, iy_l, u_l, v_l, fr_l = [], [], [], [], []
    gmax = 0.0
    pct = cfg.get('speed_range_percentile', 0.8)
    all_u = np.zeros((n_frames, h_c, w_c), np.float32)
    all_v = np.zeros((n_frames, h_c, w_c), np.float32)
    for i in range(n_frames):
        f = frames[i][:, r0:r1, c0:c1] * VEL_FACTOR
        all_u[i], all_v[i] = f[0], f[1]
        mag = np.sqrt(f[0] ** 2 + f[1] ** 2)
        nt = mask_crop & (mag > 1e-6)
        if nt.sum() == 0:
            continue
        s = mag[nt]; lo, hi = s.min(), s.max()
        gmax = max(gmax, hi)
        valid = nt & (mag >= lo + pct * (hi - lo))
        if valid.sum() == 0:
            continue
        vy, vx = np.where(valid)
        ix_l.append(vx); iy_l.append(vy)
        u_l.append(f[0, vy, vx]); v_l.append(f[1, vy, vx]); fr_l.append(np.full(len(vx), i))
    sp_ix = np.concatenate(ix_l); sp_iy = np.concatenate(iy_l)
    sp_u = np.concatenate(u_l); sp_v = np.concatenate(v_l); sp_fr = np.concatenate(fr_l)
    U = float(gmax)

    iy, ix = np.where(mask_crop)
    xt = (ix - cx_px) * PIXEL_SIZE_M / L
    yt = (iy - cy_px) * PIXEL_SIZE_M / L
    interior = np.stack([xt, yt], 1)
    cl_tol = 4.0 * PIXEL_SIZE_M / L
    cl_pts = interior[np.abs(yt) < cl_tol]
    cnts, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).squeeze().astype(float)
    return dict(mask_crop=mask_crop, crop=(r0, r1, c0, c1), h_crop=h_c, w_crop=w_c,
                cx_px=cx_px, cy_px=cy_px, L=L, L_full=L_full, U=U, n_frames=n_frames,
                sparse_ix=sp_ix, sparse_iy=sp_iy, sparse_frame=sp_fr,
                sparse_u=sp_u, sparse_v=sp_v,
                sparse_xt=(sp_ix - cx_px) * PIXEL_SIZE_M / L,
                sparse_yt=(sp_iy - cy_px) * PIXEL_SIZE_M / L,
                sparse_u_hat=sp_u / U, sparse_v_hat=sp_v / U,
                interior_coords=interior, interior_ix=ix, interior_iy=iy,
                cl_pts=cl_pts, cl_x_min=float(cl_pts[:, 0].min()), cl_x_max=float(cl_pts[:, 0].max()),
                bnd_cnt_px=cnt, all_u=all_u, all_v=all_v, data_dir=data_dir)


def site_keys(data):
    """3-px site groups pooling a location, its mirror twin and all temporal repeats
    (same rule as the original hold-out evaluation scripts)."""
    # exact float expression of the original split (tie-breaking matters: 35,764 val pts),
    # always evaluated with the full-L normalisation so every model gets the same split
    Lf = data['L_full']
    _px = PIXEL_SIZE_M / Lf
    xt = (data['sparse_ix'] - data['cx_px']) * PIXEL_SIZE_M / Lf
    yt = (data['sparse_iy'] - data['cy_px']) * PIXEL_SIZE_M / Lf
    qx = np.round(xt / (3 * _px)).astype(np.int64)
    qy = np.round(np.abs(yt) / (3 * _px)).astype(np.int64)
    return qx * 1000003 + qy


def holdout_split(data, seed=42):
    keys = site_keys(data)
    uniq, inv = np.unique(keys, return_inverse=True)
    perm = np.random.RandomState(seed).permutation(len(uniq))
    gtrain = np.zeros(len(uniq), bool)
    gtrain[perm[:int(0.8 * len(uniq))]] = True
    return np.where(gtrain[inv])[0], np.where(~gtrain[inv])[0], inv, len(uniq)


def occlusion_split(data, frac):
    """Hide the arc-side fraction `frac` of observation sites (largest x; the arc is at +x)."""
    keys = site_keys(data)
    uniq, inv = np.unique(keys, return_inverse=True)
    site_x = np.zeros(len(uniq)); np.maximum.at(site_x, inv, data['sparse_xt'])
    order = np.argsort(-site_x)
    n_hide = int(round(frac * len(uniq)))
    hidden = np.zeros(len(uniq), bool); hidden[order[:n_hide]] = True
    x_thr = site_x[order[n_hide - 1]] if n_hide > 0 else np.inf
    return np.where(~hidden[inv])[0], np.where(hidden[inv])[0], x_thr


def create_boundary_band(mask, band_width, inner_frac, cx_px, cy_px, L):
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    sd = dist_outside - dist_inside
    by, bx = np.where(np.abs(sd) <= band_width)
    d = np.abs(sd[by, bx])
    it = band_width * inner_frac
    w = np.ones(len(d)); o = d > it
    w[o] = 1.0 - (d[o] - it) / (band_width - it)
    w = np.clip(w, 0, 1)
    gy, gx = np.gradient(sd)
    n = np.stack([gx[by, bx], gy[by, bx]], 1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    pts = np.stack([(bx - cx_px) * PIXEL_SIZE_M / L, (by - cy_px) * PIXEL_SIZE_M / L], 1)
    return pts, n, w


# ----------------------------------------------------------------------------- nets
class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features=2, out_features=128, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, (out_features - in_features) // 2) * sigma,
                              requires_grad=False)

    def forward(self, v):
        x = 2.0 * np.pi * v @ self.B
        return torch.cat([v, torch.sin(x), torch.cos(x)], -1)


class DistanceNet(nn.Module):
    def __init__(self, hidden_dims=(32, 32, 32)):
        super().__init__()
        layers, d = [], 2
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


class FlowNet(nn.Module):
    def __init__(self, hidden_dims, fourier=True, sigma=0.5):
        super().__init__()
        self.fourier = fourier
        if fourier:
            self.fourier_map = FourierFeatureMapping(2, 128, sigma)
        layers, d = [], (128 if fourier else 2)
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, xy):
        out = self.net(self.fourier_map(xy) if self.fourier else xy)
        return out[:, 0:1], out[:, 1:2]


class ModifiedMLP(nn.Module):
    """Wang, Teng & Perdikaris (2021) modified MLP (gated U/V encoders)."""
    def __init__(self, width=192, depth=4, fourier=True, sigma=0.5):
        super().__init__()
        self.fourier = fourier
        if fourier:
            self.fourier_map = FourierFeatureMapping(2, 128, sigma)
        d = 128 if fourier else 2
        self.Uenc = nn.Linear(d, width); self.Venc = nn.Linear(d, width)
        self.hid = nn.ModuleList([nn.Linear(d, width)] + [nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, xy):
        x = self.fourier_map(xy) if self.fourier else xy
        U = torch.tanh(self.Uenc(x)); V = torch.tanh(self.Venc(x))
        h = x
        for lin in self.hid:
            z = torch.tanh(lin(h))
            h = (1 - z) * U + z * V
        o = self.out(h)
        return o[:, 0:1], o[:, 1:2]


class ArcForce(nn.Module):
    def __init__(self, x_frac_init, sigma_init, amp_init, y_center, x_min, x_max):
        super().__init__()
        frac = float(np.clip(x_frac_init, 0.01, 0.99))
        self.raw_x = nn.Parameter(torch.tensor(np.log(frac / (1 - frac)), dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(float(sigma_init)), dtype=torch.float32))
        self.log_amp = nn.Parameter(torch.tensor(np.log(float(amp_init)), dtype=torch.float32))
        self.y_center, self.x_min, self.x_max = float(y_center), float(x_min), float(x_max)

    def _x_arc(self):
        return self.x_min + (self.x_max - self.x_min) * torch.sigmoid(self.raw_x)

    def forward(self, xy):
        xa = self._x_arc(); s = torch.exp(self.log_sigma); a = torch.exp(self.log_amp)
        dx = xy[:, 0:1] - xa; dy = xy[:, 1:2] - self.y_center
        env = a * torch.exp(-(dx ** 2 + dy ** 2) / (2 * s ** 2))
        return env * dx / s ** 2, env * dy / s ** 2

    def get_params(self):
        with torch.no_grad():
            return self._x_arc().item(), torch.exp(self.log_sigma).item(), torch.exp(self.log_amp).item()


class Recon:
    """Everything needed to evaluate a trained reconstruction."""
    def __init__(self, model, dist_net, arc, log_nu, log_kappa, cfg, data):
        self.model, self.dist_net, self.arc = model, dist_net, arc
        self.log_nu, self.log_kappa, self.cfg, self.data = log_nu, log_kappa, cfg, data

    def psi_p(self, xy):
        psi_raw, p = self.model(xy)
        if self.cfg['hard_bc']:
            psi_raw = self.dist_net(xy) ** 2 * psi_raw
        return psi_raw, p

    def vel_hat(self, xy, create_graph=True):
        """depth-averaged dimensionless velocity (network space)."""
        xy = xy.clone().requires_grad_(True)
        psi, p = self.psi_p(xy)
        g = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=create_graph)[0]
        return g[:, 1:2], -g[:, 0:1], p, xy

    def nu_hat(self):
        return torch.exp(self.log_nu)

    def kappa_hat(self):
        return torch.exp(self.log_kappa) if self.log_kappa is not None else None

    @torch.enable_grad()
    def predict_surface_SI(self, xy_np, batch=5000):
        """surface velocity (m/s) at normalised coords -- what is compared with observations."""
        U, sf = self.data['U'], self.cfg['surface_factor']
        out_u, out_v = [], []
        for s in range(0, len(xy_np), batch):
            xy = torch.tensor(xy_np[s:s + batch], dtype=torch.float32, device=DEVICE)
            u, v, _, _ = self.vel_hat(xy, create_graph=False)
            out_u.append(u.detach().cpu().numpy().ravel() * U * sf)
            out_v.append(v.detach().cpu().numpy().ravel() * U * sf)
        return np.concatenate(out_u), np.concatenate(out_v)

    @torch.enable_grad()
    def fields_SI(self):
        d = self.data
        h, w = d['h_crop'], d['w_crop']
        U, L, sf = d['U'], d['L'], self.cfg['surface_factor']
        ix, iy = d['interior_ix'], d['interior_iy']
        u = np.zeros((h, w), np.float32); v = np.zeros_like(u); p = np.zeros_like(u)
        for s in range(0, len(ix), 5000):
            bx, by = ix[s:s + 5000], iy[s:s + 5000]
            xy = torch.tensor(np.stack([(bx - d['cx_px']) * PIXEL_SIZE_M / L,
                                        (by - d['cy_px']) * PIXEL_SIZE_M / L], 1),
                              dtype=torch.float32, device=DEVICE)
            uu, vv, pp, _ = self.vel_hat(xy, create_graph=False)
            u[by, bx] = uu.detach().cpu().numpy().ravel() * U * sf
            v[by, bx] = vv.detach().cpu().numpy().ravel() * U * sf
            p[by, bx] = pp.detach().cpu().numpy().ravel() * RHO * U ** 2
        return u, v, p


# ----------------------------------------------------------------------------- residuals
def _derivs(rec, xy_flat):
    psi, p = rec.psi_p(xy_flat)
    ones = torch.ones_like(psi)
    g = torch.autograd.grad(psi, xy_flat, ones, create_graph=True, retain_graph=True)[0]
    u, v = g[:, 1:2], -g[:, 0:1]
    gp = torch.autograd.grad(p, xy_flat, ones, create_graph=True, retain_graph=True)[0]
    gu = torch.autograd.grad(u, xy_flat, ones, create_graph=True, retain_graph=True)[0]
    gv = torch.autograd.grad(v, xy_flat, ones, create_graph=True, retain_graph=True)[0]
    return u, v, gp, gu, gv


def _forcing(rec, xy, u, v):
    cfg = rec.cfg
    if cfg.get('disable_arc_force', False):
        fx = torch.zeros_like(u); fy = torch.zeros_like(u)
    else:
        fx, fy = rec.arc(xy)
    if rec.log_kappa is not None:           # linear drag -kappa u (moved to the force side)
        k = rec.kappa_hat()
        fx = fx - k * u; fy = fy - k * v
    return fx, fy


def weak_residual(rec, xy_c, radius=None, n_mc=None, gen=None):
    cfg = rec.cfg
    nu = rec.nu_hat()
    Nc = xy_c.shape[0]
    N_mc = n_mc or cfg['mc_samples']; R = radius or cfg['mc_radius']
    r = R * torch.sqrt(torch.rand(Nc, N_mc, 1, device=xy_c.device, generator=gen))
    th = 2 * np.pi * torch.rand(Nc, N_mc, 1, device=xy_c.device, generator=gen)
    q = xy_c.unsqueeze(1) + torch.cat([r * torch.cos(th), r * torch.sin(th)], -1)
    xf = q.view(-1, 2); xf.requires_grad_(True)
    u, v, gp, gu, gv = _derivs(rec, xf)
    fx, fy = _forcing(rec, xf, u, v)
    sh = lambda t: t.view(Nc, N_mc, 1)
    R2 = R ** 2
    dx = q[:, :, 0:1] - xy_c[:, None, 0:1]; dy = q[:, :, 1:2] - xy_c[:, None, 1:2]
    r2 = dx ** 2 + dy ** 2; m = (r2 <= R2).float()
    phi = (1 - r2 / R2) ** 2 * m
    phx = -4.0 / R2 * dx * (1 - r2 / R2) * m
    phy = -4.0 / R2 * dy * (1 - r2 / R2) * m
    U_, V_ = sh(u), sh(v)
    ux, uy, vx, vy = sh(gu[:, 0:1]), sh(gu[:, 1:2]), sh(gv[:, 0:1]), sh(gv[:, 1:2])
    cx = U_ * ux + V_ * uy + sh(gp[:, 0:1]) - sh(fx)
    cy = U_ * vx + V_ * vy + sh(gp[:, 1:2]) - sh(fy)
    rx = (cx * phi + nu * (ux * phx + uy * phy)).mean(1)
    ry = (cy * phi + nu * (vx * phx + vy * phy)).mean(1)
    return rx, ry


def strong_residual(rec, xy):
    nu = rec.nu_hat()
    xy = xy.clone().requires_grad_(True)
    u, v, gp, gu, gv = _derivs(rec, xy)
    ones = torch.ones_like(u)
    uxx = torch.autograd.grad(gu[:, 0:1], xy, ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(gu[:, 1:2], xy, ones, create_graph=True, retain_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(gv[:, 0:1], xy, ones, create_graph=True, retain_graph=True)[0][:, 0:1]
    vyy = torch.autograd.grad(gv[:, 1:2], xy, ones, create_graph=True, retain_graph=True)[0][:, 1:2]
    fx, fy = _forcing(rec, xy, u, v)
    rx = u * gu[:, 0:1] + v * gu[:, 1:2] + gp[:, 0:1] - nu * (uxx + uyy) - fx
    ry = u * gv[:, 0:1] + v * gv[:, 1:2] + gp[:, 1:2] - nu * (vxx + vyy) - fy
    return rx, ry


# ----------------------------------------------------------------------------- ConFIG
def config_direction(grads):
    """ConFIG (Liu, Chu & Thuerey 2025): equal-projection direction, projection length."""
    G = torch.stack(grads)
    units = torch.nan_to_num(G / G.norm(dim=1, keepdim=True), 0)
    w = torch.ones(G.shape[0], device=G.device)
    d = torch.linalg.lstsq(units, w.unsqueeze(1)).solution.squeeze(1)
    du = d / (d.norm() + 1e-20)
    return (G @ du).sum() * du


def flat_grad(loss, params, retain=True):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


# ----------------------------------------------------------------------------- train
def run(cfg, run_dir, verbose=True):
    os.makedirs(run_dir, exist_ok=True)
    seed = cfg['seed']
    torch.manual_seed(seed); np.random.seed(seed)
    t_start = time.time()
    data = load_data(cfg)
    U, L = data['U'], data['L']

    # observation subset (holdout / occlusion)
    n_all = len(data['sparse_xt'])
    train_idx = np.arange(n_all); val_idx = np.array([], int); hidden_idx = np.array([], int)
    x_thr = None
    if cfg['holdout']:
        train_idx, val_idx, _, n_sites = holdout_split(data, 42)
        if seed == 42:
            # the original split reseeds the global RNG to 42 and draws the
            # site permutation from it; replicate so training sampling matches.
            np.random.seed(42); np.random.permutation(n_sites)
    if cfg['occlude_frac'] > 0:
        keep, hidden_idx, x_thr = occlusion_split(data, cfg['occlude_frac'])
        train_idx = np.intersect1d(train_idx, keep)
    obs_xt = data['sparse_xt'][train_idx]; obs_yt = data['sparse_yt'][train_idx]
    obs_u = data['sparse_u_hat'][train_idx]; obs_v = data['sparse_v_hat'][train_idx]
    n_obs = len(obs_xt)

    # networks (same RNG order as the original scripts)
    if cfg['backbone'] == 'modified_mlp':
        model = ModifiedMLP(fourier=cfg['fourier'], sigma=cfg['fourier_sigma']).to(DEVICE)
    else:
        model = FlowNet(cfg['hidden_dims'], cfg['fourier'], cfg['fourier_sigma']).to(DEVICE)
    dist_net, dist_mse = None, None
    if cfg['hard_bc']:
        dist_net = DistanceNet().to(DEVICE)
        din = distance_transform_edt(data['mask_crop']) * PIXEL_SIZE_M / L
        iy, ix = np.where(data['mask_crop'])
        xy_t = torch.tensor(np.stack([(ix - data['cx_px']) * PIXEL_SIZE_M / L,
                                      (iy - data['cy_px']) * PIXEL_SIZE_M / L], 1),
                            dtype=torch.float32, device=DEVICE)
        d_t = torch.tensor(din[iy, ix][:, None], dtype=torch.float32, device=DEVICE)
        od = torch.optim.Adam(dist_net.parameters(), lr=1e-3)
        for _ in range(cfg['dist_steps']):
            od.zero_grad(); ld = torch.nn.functional.mse_loss(dist_net(xy_t), d_t); ld.backward(); od.step()
        dist_mse = float(ld.item())
        for p in dist_net.parameters():
            p.requires_grad = False
    arc = ArcForce(cfg['arc_x_frac_init'], cfg['arc_sigma_init_m'] / L, cfg['arc_amp_init'],
                   0.0, data['cl_x_min'], data['cl_x_max']).to(DEVICE)

    # viscosity: learnable (default) or fixed physical value (cao_mode)
    if cfg['fixed_nu_phys'] is not None:
        log_nu = torch.tensor(np.log(cfg['fixed_nu_phys'] / (U * L)), dtype=torch.float32, device=DEVICE)
    else:
        log_nu = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32, device=DEVICE))
    log_kappa = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32, device=DEVICE)) if cfg['drag'] else None
    rec = Recon(model, dist_net, arc, log_nu, log_kappa, cfg, data)

    all_int = torch.tensor(data['interior_coords'], dtype=torch.float32, device=DEVICE)
    cl_pts = torch.tensor(data['cl_pts'], dtype=torch.float32, device=DEVICE)
    if not cfg['hard_bc']:
        bp, bn, bw = create_boundary_band(data['mask_crop'], cfg['boundary_band_width'],
                                          cfg['boundary_inner_fraction'], data['cx_px'], data['cy_px'], L)
        bnd_p = torch.tensor(bp, dtype=torch.float32, device=DEVICE)
        bnd_n = torch.tensor(bn, dtype=torch.float32, device=DEVICE)
        bnd_w = torch.tensor(bw, dtype=torch.float32, device=DEVICE)
    p_ref = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    sig_t = cfg['arc_sigma_init_m'] / L
    xa_t = data['cl_x_min'] + cfg['arc_x_frac_measured'] * (data['cl_x_max'] - data['cl_x_min'])
    xa_tol = cfg['arc_x_frac_tolerance'] * (data['cl_x_max'] - data['cl_x_min'])

    phys = [log_nu] if isinstance(log_nu, nn.Parameter) else []
    if log_kappa is not None:
        phys.append(log_kappa)
    groups = [{'params': list(model.parameters()), 'lr': cfg['lr']}]
    if phys:
        groups.append({'params': phys, 'lr': cfg['lr_physics']})
    if not cfg['disable_arc_force']:
        groups.append({'params': list(arc.parameters()), 'lr': cfg['lr_physics']})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['n_epochs'])
    # the Fourier matrix B is a frozen nn.Parameter -> exclude from explicit gradient ops
    all_params = [p for g in groups for p in g['params'] if p.requires_grad]
    net_params = [p for p in model.parameters() if p.requires_grad]

    E = torch.ones(all_int.shape[0], dtype=torch.float32, device=DEVICE)
    lam_data = cfg['lambda_data']
    keys = ['total', 'data', 'boundary', 'pde', 'p_ref', 'symmetry', 'arc_prior', 'nu_hat',
            'nu_phys', 'arc_x', 'arc_sigma', 'arc_amp', 'kappa_hat', 'lam_data', 'E_mean']
    hist = {k: [] for k in keys}
    sf = cfg['surface_factor']
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t_train0 = time.time()

    for ep in range(cfg['n_epochs']):
        opt.zero_grad()
        n_s = min(cfg['n_obs_per_step'], n_obs)
        sel = np.random.choice(n_obs, n_s, replace=False)
        xy_d = torch.tensor(np.stack([obs_xt[sel], obs_yt[sel]], 1), dtype=torch.float32, device=DEVICE)
        up, vp, _, _ = rec.vel_hat(xy_d)
        up = up * sf; vp = vp * sf
        ut = torch.tensor(obs_u[sel], dtype=torch.float32, device=DEVICE)
        vt = torch.tensor(obs_v[sel], dtype=torch.float32, device=DEVICE)
        so = torch.sqrt(ut ** 2 + vt ** 2); ws = so / (so.mean() + 1e-8)
        L_data = (ws * (up.squeeze() - ut) ** 2).mean() + (ws * (vp.squeeze() - vt) ** 2).mean()
        spd = torch.sqrt(up.squeeze() ** 2 + vp.squeeze() ** 2)
        L_data = L_data + cfg['lambda_peak'] * (so.max() - spd.max()) ** 2

        L_bnd = torch.tensor(0.0, device=DEVICE)
        if not cfg['hard_bc'] and cfg['lambda_boundary'] > 0:
            nb = min(cfg['n_boundary_points'], bnd_p.shape[0])
            ib = torch.randperm(bnd_p.shape[0])[:nb]
            ub, vb, _, _ = rec.vel_hat(bnd_p[ib])
            un = ub.squeeze() * bnd_n[ib, 0] + vb.squeeze() * bnd_n[ib, 1]
            L_bnd = (bnd_w[ib] * un ** 2).mean()

        nc = min(cfg['n_pde_per_step'], all_int.shape[0])
        if cfg['rar']:
            c = cfg['rar_c'] if cfg['rar_c_rel'] is None else cfg['rar_c_rel'] * E.mean()
            pr = E + c; pr = pr / pr.sum()
            ic = torch.multinomial(pr, nc, replacement=False)
        else:
            ic = torch.randperm(all_int.shape[0])[:nc]
        if cfg['form'] == 'weak':
            rx, ry = weak_residual(rec, all_int[ic])
        else:
            rx, ry = strong_residual(rec, all_int[ic])
        per = rx ** 2 + ry ** 2
        L_pde = per.mean()
        if cfg['rar']:
            with torch.no_grad():
                a = cfg['rar_alpha']
                E[ic] = a * E[ic] + (1 - a) * per.squeeze()

        _, _, pr_, _ = rec.vel_hat(p_ref)
        L_p = pr_.squeeze() ** 2
        L_sym = torch.tensor(0.0, device=DEVICE)
        if cfg['lambda_symmetry'] > 0 and cl_pts.shape[0] > 0:
            icl = torch.randperm(cl_pts.shape[0])[:min(200, cl_pts.shape[0])]
            _, vcl, _, _ = rec.vel_hat(cl_pts[icl])
            L_sym = (vcl ** 2).mean()
        L_prior = torch.tensor(0.0, device=DEVICE)
        if not cfg['disable_arc_force']:
            if cfg['lambda_arc_sigma_prior'] > 0:
                L_prior = L_prior + cfg['lambda_arc_sigma_prior'] * (arc.log_sigma - np.log(sig_t)) ** 2 \
                    / (2 * cfg['arc_sigma_prior_log_std'] ** 2)
            if cfg['lambda_arc_pos_prior'] > 0:
                L_prior = L_prior + cfg['lambda_arc_pos_prior'] * ((arc._x_arc() - xa_t) / xa_tol) ** 2
        reg = cfg['lambda_p_ref'] * L_p + cfg['lambda_symmetry'] * L_sym + L_prior

        if cfg['balancing'] == 'config':
            objs = [L_data, L_pde] + ([L_bnd] if not cfg['hard_bc'] else [])
            gs = [flat_grad(o, all_params) for o in objs]
            g = config_direction(gs) + flat_grad(reg, all_params, retain=False)
            off = 0
            for p in all_params:
                n = p.numel(); p.grad = g[off:off + n].view_as(p).clone(); off += n
            loss = sum(objs) + reg
        else:
            if cfg['balancing'] == 'lra' and ep % 10 == 0:
                gp_ = flat_grad(L_pde, net_params); gd_ = flat_grad(L_data, net_params)
                lam_hat = (gp_.abs().max() / (gd_.abs().mean() + 1e-12)).item()
                lam_data = 0.9 * lam_data + 0.1 * lam_hat
            loss = (lam_data * L_data + cfg['lambda_pde'] * L_pde +
                    (cfg['lambda_boundary'] * L_bnd if not cfg['hard_bc'] else 0.0) + reg)
            loss.backward()
        opt.step(); sched.step()

        nuv = rec.nu_hat().item()
        xa, sa, aa = arc.get_params()
        for k, val in [('total', loss.item()), ('data', L_data.item()), ('boundary', L_bnd.item()),
                       ('pde', L_pde.item()), ('p_ref', L_p.item()), ('symmetry', L_sym.item()),
                       ('arc_prior', L_prior.item()), ('nu_hat', nuv), ('nu_phys', nuv * U * L),
                       ('arc_x', xa), ('arc_sigma', sa), ('arc_amp', aa),
                       ('kappa_hat', rec.kappa_hat().item() if log_kappa is not None else np.nan),
                       ('lam_data', lam_data), ('E_mean', E.mean().item())]:
            hist[k].append(val)
        if verbose and (ep + 1) % 200 == 0:
            print(f"  ep {ep+1} loss={loss.item():.4f} data={L_data.item():.4f} "
                  f"pde={L_pde.item():.5f} nu_hat={nuv:.5f}", flush=True)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t_train = time.time() - t_train0
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE == 'cuda' else None

    # ---- summary ---------------------------------------------------------------
    t_inf0 = time.time()
    u_f, v_f, p_f = rec.fields_SI()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t_inf = time.time() - t_inf0
    mask = data['mask_crop']
    speed = np.sqrt(u_f ** 2 + v_f ** 2)
    nuv = rec.nu_hat().item(); nu_phys = nuv * U * L
    xa, sa, aa = arc.get_params()
    props = dict(
        cfg=cfg, U=U, L=L, L_full=data['L_full'], n_obs_train=int(n_obs), n_obs_total=int(n_all),
        nu_hat=nuv, nu_phys=nu_phys, mu_mPas=nu_phys * RHO * 1e3, Re=1.0 / nuv,
        kappa_phys=(rec.kappa_hat().item() * U / L) if log_kappa is not None else None,
        u_max=float(speed[mask].max()), u_mean=float(speed[mask].mean()),
        arc_x_tilde=xa, arc_x_mm=xa * L * 1e3, arc_sigma_mm=sa * L * 1e3,
        A_Nm3_as_reported=aa * RHO * U ** 2 / L, A_tilde=aa,   # same convention as physical_properties.json
        F_peak_Nm3=aa * RHO * U ** 2 / L / sa * math.exp(-0.5),  # true peak force density
        p_range_Pa=float(p_f[mask].max() - p_f[mask].min()),
        final_losses={k: hist[k][-1] for k in ['total', 'data', 'boundary', 'pde', 'p_ref', 'symmetry', 'arc_prior']},
        pde_last50_mean=float(np.mean(hist['pde'][-50:])),
        dist_mse=dist_mse, E_final_mean=float(E.mean().item()), E_final_min=float(E.min().item()),
        E_final_max=float(E.max().item()),
        train_time_s=t_train, total_time_s=time.time() - t_start, inference_time_s=t_inf,
        peak_mem_MB=peak_mem, gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu',
        n_params=int(sum(p.numel() for p in model.parameters())),
    )
    if props['kappa_phys'] and cfg['fixed_nu_phys']:
        props['h_eff_mm'] = math.sqrt(3 * cfg['fixed_nu_phys'] / props['kappa_phys']) * 1e3
    np.savez_compressed(os.path.join(run_dir, 'fields.npz'), u_ms=u_f, v_ms=v_f, p_Pa=p_f, mask=mask)
    np.savez_compressed(os.path.join(run_dir, 'history.npz'), **{k: np.array(v) for k, v in hist.items()})
    torch.save({'model': model.state_dict(), 'dist': dist_net.state_dict() if dist_net else None,
                'arc': arc.state_dict(), 'log_nu': log_nu.detach().cpu(),
                'log_kappa': log_kappa.detach().cpu() if log_kappa is not None else None, 'cfg': cfg},
               os.path.join(run_dir, 'model.pt'))
    split = dict(train_idx=train_idx, val_idx=val_idx, hidden_idx=hidden_idx, x_thr=x_thr)
    return rec, data, props, split
