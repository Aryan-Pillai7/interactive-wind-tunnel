"""Metrics computed from flow fields. (Owner: Aryan)

The same functions are applied to ground-truth and predicted fields so every
comparison uses the same estimator. Fields are [3, NY, NX] (u, v, p) in the
contract's units; mask is bool [NY, NX], True inside the obstacle.
"""

import numpy as np
from scipy.ndimage import binary_dilation

from windtunnel import contract as C
from windtunnel.geometry import obstacle_bounds, obstacle_height


def relative_l2(pred, true, mask):
    """Per-channel relative L2 error over fluid cells: ||pred - true|| / ||true||.

    Returns dict {u, v, p, vel}; vel uses the stacked (u, v) vector. A channel
    whose reference is (numerically) zero gives NaN rather than a huge ratio.
    """
    fluid = ~np.asarray(mask, bool)
    err = np.asarray(pred, np.float64)[:, fluid] - np.asarray(true, np.float64)[:, fluid]
    ref = np.asarray(true, np.float64)[:, fluid]
    def ratio(e, r):
        r = np.linalg.norm(r)
        return float(np.linalg.norm(e) / r) if r > 1e-9 else float("nan")

    out = {name: ratio(err[c], ref[c]) for c, name in enumerate(C.TARGET_CHANNELS)}
    out["vel"] = ratio(err[:2], ref[:2])
    return out


def divergence(u, v):
    """Central-difference du/dx + dv/dy, float [NY, NX]."""
    return np.gradient(u, axis=1) + np.gradient(v, axis=0)


def vorticity(u, v):
    """Central-difference dv/dx - du/dy, float [NY, NX]."""
    return np.gradient(v, axis=1) - np.gradient(u, axis=0)


def interior(mask):
    """Fluid cells whose central-difference stencil touches no solid (obstacle, walls, edges)."""
    solid = np.asarray(mask, bool).copy()
    solid[0, :] = solid[-1, :] = True
    out = ~binary_dilation(solid)
    out[:, 0] = out[:, -1] = False
    return out


def mean_sq_divergence(u, v, mask):
    """Mean of (du/dx + dv/dy)^2 over interior fluid cells (the loss's physics term)."""
    return float(np.mean(divergence(u, v)[interior(mask)] ** 2))


def pressure_forces(p, mask, d=None):
    """Pressure drag and lift coefficients by integrating p over the surface.

    The staircase surface is the set of faces between obstacle cells and
    fluid cells; each face (length 1) carries the pressure of its fluid cell
    and an outward normal along +-x or +-y. Force on the body = -sum p n.
    p is already a coefficient (divided by 0.5 U0^2), so C = F / D.
    Returns (cd_p, cl_p). Pressure only; never compare to cd_total.
    """
    m = np.asarray(mask, bool)
    fl = ~m
    p = np.asarray(p, np.float64)
    d = obstacle_height(m) if d is None else d
    fx = (-p[:, 1:][m[:, :-1] & fl[:, 1:]].sum()      # face normal +x
          + p[:, :-1][m[:, 1:] & fl[:, :-1]].sum())    # face normal -x
    fy = (-p[1:, :][m[:-1, :] & fl[1:, :]].sum()       # +y
          + p[:-1, :][m[1:, :] & fl[:-1, :]].sum())    # -y
    return float(fx / d), float(fy / d)


def surface_cells(mask):
    """Fluid cells 4-adjacent to the obstacle."""
    m = np.asarray(mask, bool)
    return binary_dilation(m) & ~m


def stagnation_pressure(p, mask):
    """Peak p on the fluid cells adjacent to the obstacle."""
    return float(np.asarray(p)[surface_cells(mask)].max())


def wake_window(mask):
    """Bool [NY, NX]: fluid cells from the obstacle's trailing edge to 2 D
    downstream, spanning the obstacle's height plus D/2 above and below."""
    m = np.asarray(mask, bool)
    d = obstacle_height(m)
    x0, x1, y0, y1 = obstacle_bounds(m)
    win = np.zeros_like(m)
    ya, yb = max(1, int(y0 - d / 2)), min(m.shape[0] - 1, int(y1 + d / 2) + 1)
    win[ya:yb, x1 + 1:min(m.shape[1], int(x1 + 1 + 2 * d))] = True
    return win & ~m


def wake_vorticity(u, v, mask):
    """Mean |vorticity| in the wake window, made dimensionless as |w| D / U0."""
    w = np.abs(vorticity(np.asarray(u, np.float64), np.asarray(v, np.float64)))
    return float(w[wake_window(mask)].mean() * obstacle_height(mask) / C.U0)


def field_metrics(fields, mask):
    """All derived quantities for one field, as used by evaluate.py and the app."""
    u, v, p = fields
    cd_p, cl_p = pressure_forces(p, mask)
    return {"cd_p": cd_p, "cl_p": cl_p, "stagnation_p": stagnation_pressure(p, mask),
            "wake_vorticity": wake_vorticity(u, v, mask),
            "div_ms": mean_sq_divergence(u, v, mask)}
