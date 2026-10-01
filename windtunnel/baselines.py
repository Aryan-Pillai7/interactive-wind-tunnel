"""Baselines the model is compared against. (Owner: Aryan)"""


class MeanFieldBaseline:
    """Training-set average field, masked by the query obstacle."""

    def fit(self, inputs, targets):
        raise NotImplementedError

    def predict(self, inputs):
        """inputs [3, NY, NX] raw -> fields [3, NY, NX]."""
        raise NotImplementedError


class NearestNeighbourBaseline:
    """Target of the closest training input (L2 on the raw input channels)."""

    def fit(self, inputs, targets):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError


def potential_flow_circle(mask, radius, cx, cy):
    """Analytical inviscid flow around a circle in an unbounded uniform stream.

    Returns fields [3, NY, NX] (u, v, p), 0 inside the obstacle. Circles only.

    Assumes unbounded flow; our channel has up to ~26% blockage at max D
    (D_MAX = 16), which invalidates it as a true null, but it is still useful
    as a trivial baseline. Let the numbers speak.
    """
    raise NotImplementedError
