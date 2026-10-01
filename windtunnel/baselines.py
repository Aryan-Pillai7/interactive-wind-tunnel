"""Baselines the model is compared against. (Owner: Aryan)

All take RAW inputs [3, NY, NX] (as stored in dataset.npz) and return fields
[3, NY, NX] in the contract's units, 0 inside the obstacle.
"""

import numpy as np

from windtunnel import contract as C
from windtunnel.geometry import obstacle_height


class MeanFieldBaseline:
    """Training-set average field, masked by the query obstacle."""

    name = "mean_field"

    def fit(self, inputs, targets):
        self.mean = np.asarray(targets, np.float64).mean(axis=0).astype(np.float32)
        return self

    def predict(self, inputs):
        return self.mean * (1.0 - inputs[C.IN_MASK])


class NearestNeighbourBaseline:
    """Target of the closest training input.

    Distance: RMS over all cells and channels of the inputs normalised with
    the training-set channel stats (norm.json), so the SDF, mask and Re planes
    contribute on comparable scales. The returned field is re-masked by the
    query obstacle. `distance` also serves the near-duplicate leakage report.
    """

    name = "nearest_neighbour"

    def __init__(self, norm):
        self.mean = np.array(norm["inputs"]["mean"], np.float32)[:, None, None]
        self.std = np.array(norm["inputs"]["std"], np.float32)[:, None, None]

    def _feat(self, inputs):
        x = (np.asarray(inputs, np.float32) - self.mean) / self.std
        return x.reshape(x.shape[0], -1)

    def fit(self, inputs, targets):
        self.x = self._feat(inputs)
        self.x_sq = (self.x.astype(np.float64) ** 2).sum(axis=1)
        self.targets = np.asarray(targets, np.float32)
        return self

    def query(self, inputs, exclude_self=False):
        """(index, rms distance) of the nearest training sample for each input in a batch."""
        q = self._feat(inputs)
        d2 = (q.astype(np.float64) ** 2).sum(axis=1)[:, None] + self.x_sq[None] \
            - 2.0 * (q @ self.x.T).astype(np.float64)
        if exclude_self:
            d2[d2 < 1e-6 * q.shape[1]] = np.inf
        idx = d2.argmin(axis=1)
        dist = np.sqrt(np.maximum(d2[np.arange(len(idx)), idx], 0.0) / q.shape[1])
        return idx, dist

    def predict(self, inputs):
        idx, _ = self.query(inputs[None])
        return self.targets[idx[0]] * (1.0 - inputs[C.IN_MASK])


def potential_flow_circle(mask, radius=None, cx=C.OBSTACLE_CX, cy=C.OBSTACLE_CY):
    """Analytical inviscid flow around a circle in an unbounded uniform stream.

    Returns fields [3, NY, NX] (u, v, p), 0 inside the obstacle. Circles only.
    p is the Bernoulli coefficient 1 - |U|^2 / U0^2, i.e. referenced to the
    free stream, whereas the LBM p is referenced to the outlet.

    Assumes unbounded flow; our channel has up to ~26% blockage at max D
    (D_MAX = 16), which invalidates it as a true null, but it is still useful
    as a trivial baseline. Let the numbers speak.
    """
    mask = np.asarray(mask, bool)
    radius = obstacle_height(mask) / 2.0 if radius is None else radius
    y, x = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float64)
    x, y = x - cx, y - cy
    r2 = np.maximum(x**2 + y**2, 1e-12)
    a2 = radius**2
    u = C.U0 * (1.0 - a2 * (x**2 - y**2) / r2**2)
    v = -C.U0 * 2.0 * a2 * x * y / r2**2
    p = 1.0 - (u**2 + v**2) / C.U0**2
    return (np.stack([u, v, p]) * ~mask).astype(np.float32)


class PotentialFlowBaseline:
    """potential_flow_circle as a predictor; returns None for non-circles."""

    name = "potential_flow"

    def fit(self, inputs, targets):
        return self

    def predict(self, inputs, kind="circle"):
        if kind != "circle":
            return None
        return potential_flow_circle(inputs[C.IN_MASK] > 0.5)
