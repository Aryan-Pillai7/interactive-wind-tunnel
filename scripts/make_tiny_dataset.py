"""Tiny real dataset (default 50 converged samples) from the LBM solver, for the v0.1 prototype.

Sample i gets kind TRAIN_KINDS[i % 3] and Re RES[(i // 3) % 4], so every (kind, Re) pair
appears about equally. Size, aspect and angle are random, and shapes whose measured D falls outside [D_MIN, D_MAX] are redrawn.
Non-converged solves are dropped and replaced until N samples have converged.
The split is a plain seeded 80/10/10 shuffle: with ~4 samples per (kind, Re bucket) the
stratified split would leave val and test empty.

Usage: python scripts/make_tiny_dataset.py [--n 50] [--workers 8] [--seed 0] [--out DIR]
Writes OUT/dataset.npz, OUT/meta.json, OUT/norm.json (default OUT = DATA_DIR/tiny), the same
schema as scripts/make_fake_dataset.py, so `train.py --data OUT` works on it.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from windtunnel import contract as C
from windtunnel.geometry import make_mask, obstacle_height, sdf
from windtunnel.lbm import solve

RES = (10.0, 20.0, 30.0, 40.0)
# Same aspect ranges as scripts/generate.py.
ASPECT = {"circle": (1.0, 1.0), "ellipse": (1.2, 2.5), "rectangle": (1.0, 3.0)}
# contract.MAX_ITERS (2000) is the interactive cap; converging takes ~9400 steps on median.
MAX_ITERS = 40_000


def draw_spec(i, rng):
    """Random shape parameters whose measured D is within [D_MIN, D_MAX]."""
    kind, re = C.TRAIN_KINDS[i % 3], RES[(i // 3) % len(RES)]
    while True:
        size = float(rng.uniform(C.D_MIN, C.D_MAX))
        aspect = float(rng.uniform(*ASPECT[kind]))
        angle = 0.0 if kind == "circle" else float(rng.uniform(-90.0, 90.0))
        d = float(obstacle_height(make_mask(kind, size, aspect, angle)))
        if C.D_MIN <= d <= C.D_MAX:
            return {"kind": kind, "size": size, "aspect": aspect, "angle": angle,
                    "re": re, "d": d}


def run(spec, max_iters=MAX_ITERS):
    mask = make_mask(spec["kind"], spec["size"], spec["aspect"], spec["angle"])
    try:
        r = solve(mask, spec["re"], max_iters=max_iters)
    except ValueError as e:  # tau < TAU_MIN
        return spec, None, str(e)
    x = np.stack([sdf(mask), mask.astype(np.float32),
                  np.full(C.GRID_SHAPE, C.re_to_norm(spec["re"]), np.float32)])
    y = np.stack([r["u"], r["v"], r["p"]]).astype(np.float32)
    info = {k: r[k] for k in ("cd_total", "cl_total", "iterations", "converged", "seconds")}
    return spec, (x, y, info), None


def channel_stats(x):
    mean = x.mean(axis=(0, 2, 3))
    std = np.maximum(x.std(axis=(0, 2, 3)), 1e-8)
    return {"mean": mean.tolist(), "std": std.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--out", default=C.data_path("tiny"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    inputs, targets, meta, dropped = [], [], [], []
    attempts, t0 = 0, time.time()
    with Pool(args.workers) as pool:
        while len(meta) < args.n and attempts < args.max_attempts:
            k = min(args.n - len(meta), args.max_attempts - attempts)
            specs = [draw_spec(attempts + i, rng) for i in range(k)]
            attempts += k
            for spec, out, err in pool.imap(run, specs):
                if out is None or not out[2]["converged"]:
                    dropped.append(dict(spec, reason=err or "not converged"))
                    print(f"drop {spec['kind']} Re={spec['re']:.0f}: {err or 'not converged'}")
                    continue
                x, y, info = out
                inputs.append(x)
                targets.append(y)
                meta.append(dict(spec, blockage=spec["d"] / C.CHANNEL_HEIGHT,
                                 cd_total=float(info["cd_total"]), cl_total=float(info["cl_total"]),
                                 iterations=int(info["iterations"]), seconds=float(info["seconds"])))
                print(f"[{len(meta)}/{args.n}] {spec['kind']} Re={spec['re']:.0f} D={spec['d']:.0f} "
                      f"iters={info['iterations']} {info['seconds']:.1f}s cd={info['cd_total']:.3f}")
    wall = time.time() - t0

    n = len(meta)
    inputs = np.stack(inputs).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    perm = rng.permutation(n)
    n_val = n_test = int(round(n * C.SPLIT_FRACTIONS["val"]))
    split = {"val": np.sort(perm[:n_val]), "test": np.sort(perm[n_val:n_val + n_test]),
             "train": np.sort(perm[n_val + n_test:])}

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, C.DATASET_FILE), inputs=inputs, targets=targets,
             cd_total=np.array([m["cd_total"] for m in meta], np.float32),
             cl_total=np.array([m["cl_total"] for m in meta], np.float32),
             idx_train=split["train"], idx_val=split["val"], idx_test=split["test"],
             idx_ood=np.zeros(0, np.int64))
    with open(os.path.join(args.out, C.META_FILE), "w") as f:
        json.dump({"seed": args.seed, "n_attempted": attempts, "total_converged": n,
                   "total_dropped": len(dropped), "wall_seconds": wall, "workers": args.workers,
                   "split": "random 80/10/10 (not stratified)",
                   "split_counts": {k: len(v) for k, v in split.items()},
                   "samples": meta, "dropped": dropped}, f, indent=1)
    tr = split["train"]
    with open(os.path.join(args.out, C.NORM_FILE), "w") as f:
        json.dump({"computed_on": "idx_train",
                   "inputs": dict(channels=C.INPUT_CHANNELS, **channel_stats(inputs[tr])),
                   "targets": dict(channels=C.TARGET_CHANNELS, **channel_stats(targets[tr]))},
                  f, indent=1)
    print(f"wrote {n} samples ({len(dropped)} dropped) to {args.out} in {wall:.0f}s: "
          + ", ".join(f"{k}={len(v)}" for k, v in split.items()))


if __name__ == "__main__":
    main()
