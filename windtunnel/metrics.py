"""Metrics computed from flow fields. (Owner: Aryan)

The same functions are applied to ground-truth and predicted fields so every
comparison uses the same estimator.
"""


def relative_l2(pred, true, mask):
    """Per-channel relative L2 error over fluid cells: ||pred - true|| / ||true||.

    pred, true: [3, NY, NX]; mask: bool [NY, NX]. Returns dict {u, v, p, vel},
    where vel uses the stacked (u, v) vector.
    """
    raise NotImplementedError


def divergence(u, v):
    """Central-difference du/dx + dv/dy, float [NY, NX]."""
    raise NotImplementedError


def vorticity(u, v):
    """Central-difference dv/dx - du/dy, float [NY, NX]."""
    raise NotImplementedError


def pressure_forces(p, mask, d):
    """Pressure drag and lift coefficients by integrating p over the surface.

    Returns (cd_p, cl_p). Pressure only; never compare to cd_total.
    """
    raise NotImplementedError


def stagnation_pressure(p, mask):
    """Peak p on the fluid cells adjacent to the obstacle."""
    raise NotImplementedError


def wake_vorticity(u, v, mask):
    """Mean |vorticity| in a fixed wake window behind the obstacle."""
    raise NotImplementedError
