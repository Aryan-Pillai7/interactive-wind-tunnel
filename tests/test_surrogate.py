"""Contract checks on the committed models/model.onnx (onnxruntime only, no torch)."""

import os
import time

import numpy as np
import pytest

from windtunnel import contract as C
from windtunnel import geometry
from windtunnel.dataset import build_inputs
from windtunnel.surrogate import load_surrogate

pytestmark = pytest.mark.skipif(not os.path.exists(C.MODEL_PATH), reason="no models/model.onnx")


@pytest.fixture(scope="module")
def model():
    return load_surrogate(C.MODEL_PATH)


@pytest.fixture(scope="module")
def batch():
    shapes = [("circle", 12, 1.0, 0.0, 10.0), ("ellipse", 14, 2.0, 30.0, 25.0),
              ("rectangle", 16, 2.5, -45.0, 40.0)]
    return np.stack([build_inputs(geometry.make_mask(k, s, a, g), re) for k, s, a, g, re in shapes])


def test_io_names_shapes_and_dynamic_batch(model, batch):
    out = model.predict_batch(batch)
    assert out.shape == batch.shape and out.dtype == np.float32
    assert model.predict(batch[0]).shape == (C.N_OUT, C.NY, C.NX)
    assert np.isfinite(out).all()


def test_all_channels_masked_inside_obstacle(model, batch):
    out = model.predict_batch(batch)
    inside = batch[:, C.IN_MASK] > 0.5
    for c in range(C.N_OUT):
        assert np.abs(out[:, c][inside]).max() == 0


def test_outputs_in_physical_units(model, batch):
    """Denormalisation baked in: far-field u should be O(U0), not O(1) normalised values."""
    u = model.predict_batch(batch)[:, C.OUT_U]
    far = u[:, 5:-5, 100:120]
    assert 0.2 * C.U0 < np.median(far) < 3 * C.U0


def test_latency_under_50ms(model, batch):
    model.predict(batch[0])
    t = time.perf_counter()
    for _ in range(20):
        model.predict(batch[0])
    assert (time.perf_counter() - t) / 20 * 1000 < 50
