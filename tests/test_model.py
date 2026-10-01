"""Tests for the model half (needs torch + onnx: run in the train image).

    docker compose run --rm --entrypoint python train -m pytest -q tests/test_model.py
"""

import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from windtunnel import contract as C  # noqa: E402
from windtunnel.model import Exported, build_model, param_count  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def raw_batch(n, seed=0):
    """Raw inputs with a centred circle obstacle, like the dataset."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:C.NY, 0:C.NX]
    out = np.zeros((n, C.N_IN, C.NY, C.NX), np.float32)
    for k in range(n):
        r = rng.uniform(C.D_MIN, C.D_MAX) / 2
        dist = np.hypot(x - C.OBSTACLE_CX, y - C.OBSTACLE_CY) - r
        out[k, C.IN_SDF] = dist
        out[k, C.IN_MASK] = dist <= 0
        out[k, C.IN_RE] = rng.uniform()
    return out


def exported(seed=0):
    torch.manual_seed(seed)
    stats = [[1.0, 0.1, 0.5], [2.0, 0.3, 0.3], [0.05, 0.0, 0.0], [0.01, 0.005, 0.2]]
    return Exported(build_model(), *stats).eval()


def test_param_count_in_budget():
    n = param_count(build_model())
    assert 1_000_000 <= n <= 2_000_000, n


def test_shapes_and_masking():
    x = torch.from_numpy(raw_batch(3))
    with torch.no_grad():
        y = exported()(x)
    assert y.shape == (3, C.N_OUT, C.NY, C.NX) and y.dtype == torch.float32
    inside = x[:, C.IN_MASK:C.IN_MASK + 1].expand_as(y) > 0.5
    assert torch.all(y[inside] == 0)
    assert torch.any(y[~inside] != 0)


def test_divergence_of_linear_field():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", os.path.join(REPO, "scripts", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    yy, xx = torch.meshgrid(torch.arange(C.NY, dtype=torch.float32),
                            torch.arange(C.NX, dtype=torch.float32), indexing="ij")
    u, v = (0.3 * xx)[None], (-0.1 * yy)[None]
    div = train.divergence(u, v)
    assert torch.allclose(div[:, 1:-1, 1:-1], torch.tensor(0.2), atol=1e-5)
    mask = torch.from_numpy(raw_batch(1)[:, C.IN_MASK:C.IN_MASK + 1])
    valid = train.divergence_valid(mask)[0, 0]
    assert valid[:2].sum() == 0 and valid[-2:].sum() == 0
    assert (valid * torch.nn.functional.max_pool2d(mask, 3, 1, 1)[0, 0]).sum() == 0


def test_train_never_reads_test_or_ood():
    with open(os.path.join(REPO, "scripts", "train.py")) as f:
        src = f.read()
    assert not re.search(r"idx_test|idx_ood", src.split('"""', 2)[2])


def test_onnx_export_matches_contract(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_onnx", os.path.join(REPO, "scripts", "export_onnx.py"))
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    model, path = exported(), str(tmp_path / "model.onnx")
    exp.check_io(exp.export(model, path))
    x = raw_batch(5, seed=1)
    diffs = exp.torch_vs_onnx(model, path, x)
    assert max(diffs.values()) < C.TORCH_ONNX_MAX_ABS_DIFF, diffs
    # Same call as the app/eval side, at batch 1 and batch > 1.
    sess = ort.InferenceSession(path)
    for n in (1, 4):
        out = sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: x[:n]})[0]
        assert out.shape == (n, C.N_OUT, C.NY, C.NX) and out.dtype == np.float32
