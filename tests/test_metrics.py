import numpy as np

from windtunnel import contract as C
from windtunnel import geometry, metrics


def test_uniform_flow_has_zero_divergence_and_vorticity():
    u = np.full(C.GRID_SHAPE, C.U0)
    v = np.zeros(C.GRID_SHAPE)
    assert np.abs(metrics.divergence(u, v)).max() == 0
    assert np.abs(metrics.vorticity(u, v)).max() == 0


def test_solid_rotation_vorticity_is_twice_omega():
    y, x = np.mgrid[0:C.NY, 0:C.NX].astype(float)
    om = 1e-3
    u, v = -om * (y - 32), om * (x - 64)
    assert np.allclose(metrics.vorticity(u, v), 2 * om)
    assert np.allclose(metrics.divergence(u, v), 0)


def test_uniform_pressure_gives_zero_force():
    mask = geometry.make_mask("ellipse", 14, 2.0, 30.0)
    cd, cl = metrics.pressure_forces(np.full(C.GRID_SHAPE, 3.7), mask)
    assert abs(cd) < 1e-12 and abs(cl) < 1e-12


def test_linear_pressure_on_rectangle_matches_discrete_buoyancy():
    mask = np.zeros(C.GRID_SHAPE, bool)
    mask[26:36, 28:40] = True  # H = 10 rows, W = 12 columns
    g = 0.1
    x = np.arange(C.NX)[None, :].repeat(C.NY, 0)
    cd, cl = metrics.pressure_forces(-g * x, mask)
    # front faces see x = 27, back faces x = 40: F = g * (40 - 27) * H, C = F / D
    assert np.isclose(cd, g * 13 * 10 / 10)
    assert abs(cl) < 1e-12


def test_relative_l2_zero_for_identical_and_one_for_zero_prediction():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(3,) + C.GRID_SHAPE)
    mask = geometry.make_mask("circle", 12)
    assert metrics.relative_l2(true, true, mask)["vel"] == 0
    assert np.isclose(metrics.relative_l2(np.zeros_like(true), true, mask)["p"], 1.0)
