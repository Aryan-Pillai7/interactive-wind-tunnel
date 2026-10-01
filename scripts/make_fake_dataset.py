"""Small fake dataset with the exact real schema, for pipeline testing.

Targets are analytical potential flow around circles (u, v, and Bernoulli
p = 1 - |U|^2 / U0^2, all 0 inside the obstacle). The fields do not depend
on Re and ignore the channel walls: this is for plumbing, not physics.
OOD samples are triangles with a masked uniform-flow placeholder target.
Inputs and the split use the same code as the real generator.

Usage: python scripts/make_fake_dataset.py [--n 96] [--n-ood 8] [--seed 0] [--out DIR]
Writes OUT/dataset.npz, OUT/meta.json, OUT/norm.json (default OUT = DATA_DIR/fake).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windtunnel import contract as C  # noqa: E402
from windtunnel import dataset, geometry  # noqa: E402
from windtunnel.baselines import potential_flow_circle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--n-ood", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=C.data_path("fake"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    xs, ys, samples = [], [], []
    specs = [("circle", rng.uniform(C.D_MIN, C.D_MAX)) for _ in range(args.n)]
    specs += [("triangle", rng.uniform(C.D_MIN, C.D_MAX)) for _ in range(args.n_ood)]
    for kind, size in specs:
        re = float(rng.uniform(C.RE_MIN, C.RE_MAX))
        mask = geometry.make_mask(kind, size)
        if kind == "circle":
            fields = potential_flow_circle(mask, size / 2.0)
        else:
            fields = np.zeros((C.N_OUT, C.NY, C.NX), np.float32)
            fields[C.OUT_U] = C.U0 * ~mask
        d = geometry.obstacle_height(mask)
        xs.append(dataset.build_inputs(mask, re))
        ys.append(fields)
        samples.append({"kind": kind, "params": {"size": float(size), "aspect": 1.0, "angle": 0.0},
                        "re": re, "d": d, "blockage": d / C.CHANNEL_HEIGHT,
                        "iterations": 0, "converged": True, "cd_total": 0.0, "cl_total": 0.0})

    inputs, targets = np.stack(xs), np.stack(ys)
    split, counts = dataset.split_indices(samples, rng)
    header = {"fake": True, "seed": args.seed, "n_attempted": len(samples),
              "n_kept": len(samples), "n_dropped": 0, "dropped_by_reason": {},
              "stratify_by": ["kind", "re"], "re_bucket_edges": C.RE_BUCKET_EDGES,
              "split_sizes": {k: int(len(v)) for k, v in split.items()}, "split_counts": counts}
    n = len(samples)
    dataset.write_dataset(args.out, inputs, targets, np.zeros(n), np.zeros(n), split, header,
                          samples)
    print(f"wrote {n} samples to {args.out}: {header['split_sizes']}")


if __name__ == "__main__":
    main()
