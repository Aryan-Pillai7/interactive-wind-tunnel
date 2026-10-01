"""Obstacle geometry: binary masks and signed distance fields. (Owner: Aryan)"""

import numpy as np

from windtunnel.contract import NX, NY


def make_mask(kind, size, aspect, angle, nx=NX, ny=NY):
    """Rasterise one obstacle centred at (OBSTACLE_CX, OBSTACLE_CY).

    kind: one of TRAIN_KINDS or OOD_KINDS ("circle", "ellipse", "rectangle",
        "naca", "triangle").
    size: characteristic size in cells (see the solver session for the
        exact per-kind meaning; D is measured from the mask afterwards).
    aspect: shape-specific ratio (ellipse/rectangle aspect, NACA thickness).
    angle: rotation / angle of attack in degrees, counter-clockwise.

    Returns bool [ny, nx], True inside the obstacle.
    """
    raise NotImplementedError


def sdf(mask):
    """Signed distance in cells to the obstacle surface.

    Positive in the fluid, negative inside the obstacle.
    Returns float32 [NY, NX].
    """
    raise NotImplementedError


def obstacle_height(mask):
    """D: the obstacle's extent across the flow (y), in cells."""
    raise NotImplementedError
