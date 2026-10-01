import numpy as np
import pytest

from windtunnel import contract as C
from windtunnel import geometry


@pytest.mark.parametrize("kind", geometry.KINDS)
@pytest.mark.parametrize("angle", [0.0, 37.0, -80.0])
def test_measured_height_close_to_size(kind, angle):
    for size in (C.D_MIN, C.D_MAX):
        mask = geometry.make_mask(kind, size, 1.8 if kind != "circle" else 1.0, angle)
        # A sharp triangle apex pointing across the flow can lose ~2 cells to
        # rasterisation; D is always measured from the mask, so this is harmless.
        tol = 2.5 if kind == "triangle" else 1.5
        assert abs(geometry.obstacle_height(mask) - size) <= tol


def test_sdf_sign_convention():
    mask = geometry.make_mask("circle", 12)
    s = geometry.sdf(mask)
    assert s.dtype == np.float32 and s.shape == C.GRID_SHAPE
    assert (s[mask] < 0).all() and (s[~mask] > 0).all()
    assert np.isclose(np.abs(s).min(), 0.5)


def test_symmetric_circle_is_centred_between_walls():
    mask = geometry.make_mask("circle", 12)
    assert (mask == mask[::-1]).all()
