import numpy as np
import pytest

from windtunnel import contract as C
from windtunnel import geometry, lbm


def test_rejects_small_tau():
    mask = geometry.make_mask("circle", 4)
    with pytest.raises(ValueError):
        lbm.solve(mask, 40.0, d=1.0)


def test_short_run_contract():
    mask = geometry.make_mask("circle", 12)
    r = lbm.solve(mask, 20.0, max_iters=300)
    for k in ("u", "v", "p"):
        assert r[k].shape == C.GRID_SHAPE and r[k].dtype == np.float32
        assert (r[k][mask] == 0).all()
    assert np.isclose(r["u"][1:-1, 0], C.U0).all()
    assert (r["u"][0] == 0).all() and (r["u"][-1] == 0).all()
    assert abs(r["cl_total"]) < 1e-6  # symmetric shape, symmetric channel
    assert r["cd_total"] > 0
    assert not r["converged"]
