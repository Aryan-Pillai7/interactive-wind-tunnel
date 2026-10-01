"""Small fake dataset with the exact real schema, for pipeline testing.

Targets are analytical potential flow around circles (u, v, and Bernoulli
p = 1 - |U|^2 / U0^2, all 0 inside the obstacle). The fields do not depend
on Re and ignore the channel walls: this is for plumbing, not physics.
OOD samples are triangles with a masked uniform-flow placeholder target.

Usage: python scripts/make_fake_dataset.py [--n 96] [--n-ood 8] [--seed 0] [--out DIR]
Writes OUT/dataset.npz, OUT/meta.json, OUT/norm.json (default OUT = DATA_DIR/fake).
"""

import argparse
import json
import os

import numpy as np
from scipy.ndimage import distance_transform_edt

from windtunnel import contract as C


def signed_distance(mask):
    """Positive in the fluid, negative inside (same convention as geometry.sdf)."""
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32)


def grid():
    y, x = np.mgrid[0:C.NY, 0:C.NX].astype(np.float64)
    return x - C.OBSTACLE_CX, y - C.OBSTACLE_CY


def circle_sample(radius):
    x, y = grid()
    r2 = np.maximum(x**2 + y**2, 1e-12)
    mask = r2 <= radius**2
    a2 = radius**2
    u = C.U0 * (1.0 - a2 * (x**2 - y**2) / r2**2)
    v = -C.U0 * 2.0 * a2 * x * y / r2**2
    p = 1.0 - (u**2 + v**2) / C.U0**2
    fields = np.stack([u, v, p]) * ~mask
    return mask, fields.astype(np.float32)


def triangle_sample(size):
    x, y = grid()
    # Upstream-pointing isosceles triangle: apex at -size/2, base at +size/2.
    half = size / 2.0
    mask = (x >= -half) & (x <= half) & (np.abs(y) <= (x + half) / 2.0)
    fields = np.zeros((C.N_OUT, C.NY, C.NX), np.float32)
    fields[C.OUT_U] = C.U0
    return mask, fields * ~mask


def re_bucket(re):
    return int(np.searchsorted(C.RE_BUCKET_EDGES, re, side="right") - 1)


def stratified_split(meta, rng):
    """Apply SPLIT_FRACTIONS within each (kind, Re bucket)."""
    groups = {}
    for i, m in enumerate(meta):
        if m["kind"] in C.TRAIN_KINDS:
            groups.setdefault((m["kind"], re_bucket(m["re"])), []).append(i)
    split = {k: [] for k in ("train", "val", "test")}
    counts = {}
    for key in sorted(groups):
        idx = np.array(groups[key])
        rng.shuffle(idx)
        n_val = int(round(len(idx) * C.SPLIT_FRACTIONS["val"]))
        n_test = int(round(len(idx) * C.SPLIT_FRACTIONS["test"]))
        parts = {"val": idx[:n_val], "test": idx[n_val:n_val + n_test],
                 "train": idx[n_val + n_test:]}
        for k, v in parts.items():
            split[k].extend(v.tolist())
        counts[f"{key[0]}|re_bucket={key[1]}"] = {k: len(v) for k, v in parts.items()}
    return {k: np.sort(np.array(v, np.int64)) for k, v in split.items()}, counts


def channel_stats(x):
    mean = x.mean(axis=(0, 2, 3))
    std = np.maximum(x.std(axis=(0, 2, 3)), 1e-8)
    return {"mean": mean.tolist(), "std": std.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--n-ood", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=C.data_path("fake"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    inputs, targets, meta = [], [], []
    specs = [("circle", rng.uniform(C.D_MIN, C.D_MAX)) for _ in range(args.n)]
    specs += [("triangle", rng.uniform(C.D_MIN, C.D_MAX)) for _ in range(args.n_ood)]
    for kind, d in specs:
        re = float(rng.uniform(C.RE_MIN, C.RE_MAX))
        if kind == "circle":
            mask, fields = circle_sample(d / 2.0)
        else:
            mask, fields = triangle_sample(d)
        rows = np.flatnonzero(mask.any(axis=1))
        d_meas = float(rows[-1] - rows[0] + 1)
        x = np.stack([signed_distance(mask), mask.astype(np.float32),
                      np.full(C.GRID_SHAPE, C.re_to_norm(re), np.float32)])
        inputs.append(x)
        targets.append(fields)
        meta.append({"kind": kind, "params": {"size": d, "aspect": 1.0, "angle": 0.0},
                     "re": re, "d": d_meas, "blockage": d_meas / C.CHANNEL_HEIGHT,
                     "iterations": 0, "converged": True})

    inputs = np.stack(inputs).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    split, counts = stratified_split(meta, rng)
    idx_ood = np.array([i for i, m in enumerate(meta) if m["kind"] in C.OOD_KINDS], np.int64)

    os.makedirs(args.out, exist_ok=True)
    n = len(meta)
    np.savez(os.path.join(args.out, C.DATASET_FILE), inputs=inputs, targets=targets,
             cd_total=np.zeros(n, np.float32), cl_total=np.zeros(n, np.float32),
             idx_train=split["train"], idx_val=split["val"], idx_test=split["test"],
             idx_ood=idx_ood)
    with open(os.path.join(args.out, C.META_FILE), "w") as f:
        json.dump({"fake": True, "seed": args.seed, "n_attempted": n, "n_dropped": 0,
                   "stratify_by": ["kind", "re"], "re_bucket_edges": C.RE_BUCKET_EDGES,
                   "split_counts": counts, "samples": meta}, f, indent=1)
    tr = split["train"]
    with open(os.path.join(args.out, C.NORM_FILE), "w") as f:
        json.dump({"computed_on": "idx_train",
                   "inputs": dict(channels=C.INPUT_CHANNELS, **channel_stats(inputs[tr])),
                   "targets": dict(channels=C.TARGET_CHANNELS, **channel_stats(targets[tr]))},
                  f, indent=1)
    print(f"wrote {n} samples to {args.out}: "
          + ", ".join(f"{k}={len(v)}" for k, v in split.items()) + f", ood={len(idx_ood)}")


if __name__ == "__main__":
    main()
