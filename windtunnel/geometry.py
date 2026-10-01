"""Obstacle geometry: binary masks and signed distance fields. (Owner: Aryan)

Shapes are centred at (OBSTACLE_CX, OBSTACLE_CY) and rasterised at cell
centres (cell (j, i) is the point x = i, y = j). Every shape is scaled so its
nominal extent across the flow (after rotation) equals `size`; the measured
D of the rasterised mask is within about one cell of it.

Parameters per kind:
    circle     size = diameter; aspect and angle ignored.
    ellipse    aspect = major / minor axis (>= 1); the major axis lies along
               the flow at angle 0.
    rectangle  aspect = length / thickness (>= 1); the long side lies along
               the flow at angle 0.
    triangle   isosceles, apex pointing upstream at angle 0; aspect = length
               along the flow / base. OOD only.
angle is counter-clockwise in degrees.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from windtunnel import contract as C

KINDS = C.TRAIN_KINDS + C.OOD_KINDS


def _coords(nx, ny, offset=(0.0, 0.0)):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return x - C.OBSTACLE_CX - offset[0], y - C.OBSTACLE_CY - offset[1]


def _rotate(x, y, angle):
    """Rotate points by -angle, i.e. into the body frame."""
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    return c * x + s * y, -s * x + c * y


def _polygon_mask(verts, x, y):
    """Even-odd point-in-polygon test for a convex or simple polygon."""
    inside = np.zeros(x.shape, bool)
    n = len(verts)
    for k in range(n):
        x1, y1 = verts[k]
        x2, y2 = verts[(k + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xi)
    return inside


def _scaled_polygon(body_verts, angle, size):
    """Rotate body-frame vertices by angle and scale so the y extent is size."""
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    v = np.array(body_verts, float)
    v = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)
    return v * (size / (v[:, 1].max() - v[:, 1].min()))


def make_mask(kind, size, aspect=1.0, angle=0.0, nx=C.NX, ny=C.NY, offset=(0.0, 0.0)):
    """Rasterise one obstacle. Returns bool [ny, nx], True inside.

    offset: (dx, dy) sub-cell shift of the centre, used only to measure the
    drag estimator's sensitivity to rasterisation (scripts/noise_floor.py).
    """
    x, y = _coords(nx, ny, offset)
    if kind == "circle":
        return x**2 + y**2 <= (size / 2.0) ** 2
    if kind == "ellipse":
        a = np.deg2rad(angle)
        r = size / (2.0 * np.sqrt((aspect * np.sin(a)) ** 2 + np.cos(a) ** 2))
        xb, yb = _rotate(x, y, angle)
        return (xb / (r * aspect)) ** 2 + (yb / r) ** 2 <= 1.0
    if kind == "rectangle":
        L, t = aspect / 2.0, 0.5
        verts = [(-L, -t), (L, -t), (L, t), (-L, t)]
    elif kind == "triangle":
        L = aspect / 2.0
        verts = [(-L, 0.0), (L, -0.5), (L, 0.5)]
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return _polygon_mask(_scaled_polygon(verts, angle, size), x, y)


def sdf(mask):
    """Signed distance in cells to the obstacle surface.

    The surface is taken halfway between an obstacle cell and its fluid
    neighbour, so cells touching the surface have |sdf| = 0.5.
    Positive in the fluid, negative inside the obstacle. float32 [NY, NX].
    """
    mask = np.asarray(mask, bool)
    out = distance_transform_edt(~mask) - 0.5
    inside = distance_transform_edt(mask) - 0.5
    out[mask] = -inside[mask]
    return out.astype(np.float32)


def obstacle_height(mask):
    """D: the obstacle's extent across the flow (y), in cells."""
    rows = np.flatnonzero(np.asarray(mask, bool).any(axis=1))
    return float(rows[-1] - rows[0] + 1) if rows.size else 0.0


def obstacle_bounds(mask):
    """(x_min, x_max, y_min, y_max) cell indices of the obstacle."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
