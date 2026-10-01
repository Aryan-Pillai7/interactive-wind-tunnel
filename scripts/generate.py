"""Generate the dataset with the LBM solver. (Owner: Aryan)

Usage: python scripts/generate.py --n 1600 --workers 8 --seed 0
           [--n-ood 160] [--stratify-by kind,re] [--out DIR]

--n is the number of in-distribution samples *attempted* (kinds cycle through
TRAIN_KINDS); --n-ood triangles are attempted on top (default n // 10).
Writes OUT/dataset.npz, meta.json and norm.json (schema: contract.py,
windtunnel/dataset.py). Deterministic from --seed and independent of
--workers: every sample draws its parameters from its own child seed.

Samples with tau < TAU_MIN or that do not converge within MAX_ITERS are
dropped and counted (by reason) in meta.json, never kept. Each kept sample's
blockage ratio D / CHANNEL_HEIGHT is stored in meta.json.
--stratify-by kind,re (default) splits within each (kind, Re bucket) with
SPLIT_FRACTIONS; "none" gives one plain random split. Per-bucket split counts
go to meta.json. Triangles (OOD_KINDS) only ever go to idx_ood.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windtunnel import contract as C  # noqa: E402
from windtunnel import dataset, geometry, lbm  # noqa: E402

# Aspect ranges (>= 1 means elongated along the flow at angle 0).
ASPECT = {"circle": (1.0, 1.0), "ellipse": (1.2, 2.5), "rectangle": (1.0, 3.0),
          "triangle": (0.8, 1.6)}


def sample_params(kind, seed_seq):
    rng = np.random.default_rng(seed_seq)
    size = float(rng.uniform(C.D_MIN, C.D_MAX))
    aspect = float(rng.uniform(*ASPECT[kind]))
    angle = 0.0 if kind == "circle" else float(rng.uniform(-90.0, 90.0))
    re = float(rng.uniform(C.RE_MIN, C.RE_MAX))
    return {"kind": kind, "params": {"size": size, "aspect": aspect, "angle": angle}, "re": re}


def run_one(spec):
    """Solve one sample. Returns (meta entry, (inputs, targets) or None if dropped)."""
    mask = geometry.make_mask(spec["kind"], **spec["params"])
    d = geometry.obstacle_height(mask)
    info = dict(spec, d=d, blockage=d / C.CHANNEL_HEIGHT, tau=lbm.tau_for(spec["re"], d))
    if info["tau"] < C.TAU_MIN:
        return dict(info, drop="tau"), None
    res = lbm.solve(mask, spec["re"], d=d)
    info.update(iterations=res["iterations"], converged=res["converged"],
                seconds=round(res["seconds"], 3), cd_total=res["cd_total"],
                cl_total=res["cl_total"])
    if not res["converged"]:
        return dict(info, drop="not_converged"), None
    x = dataset.build_inputs(mask, spec["re"])
    y = np.stack([res["u"], res["v"], res["p"]])
    return info, (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-ood", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify-by", default="kind,re")
    ap.add_argument("--out", default=C.DATA_DIR)
    args = ap.parse_args()
    n_ood = args.n // 10 if args.n_ood is None else args.n_ood
    stratify = () if args.stratify_by == "none" else tuple(args.stratify_by.split(","))
    if not set(stratify) <= {"kind", "re"}:
        sys.exit(f"--stratify-by must be 'kind,re', 'kind', 're' or 'none', got {args.stratify_by}")

    kinds = [C.TRAIN_KINDS[i % len(C.TRAIN_KINDS)] for i in range(args.n)]
    kinds += [C.OOD_KINDS[i % len(C.OOD_KINDS)] for i in range(n_ood)]
    seeds = np.random.SeedSequence(args.seed).spawn(len(kinds) + 1)
    specs = [sample_params(k, s) for k, s in zip(kinds, seeds[:-1])]

    t0 = time.time()
    kept, xs, ys, dropped = [], [], [], []
    with mp.Pool(args.workers) as pool:
        for k, (info, data) in enumerate(pool.imap(run_one, specs)):
            if data is None:
                dropped.append(info)
            else:
                kept.append(info)
                xs.append(data[0])
                ys.append(data[1])
            if (k + 1) % 10 == 0 or k + 1 == len(specs):
                print(f"[{k + 1}/{len(specs)}] kept {len(kept)} dropped {len(dropped)} "
                      f"{time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    if not kept:
        sys.exit("no converged samples")

    inputs, targets = np.stack(xs), np.stack(ys)
    split, counts = dataset.split_indices(kept, np.random.default_rng(seeds[-1]), stratify)
    reasons = {}
    for d in dropped:
        reasons[d["drop"]] = reasons.get(d["drop"], 0) + 1
    header = {"fake": False, "seed": args.seed, "n_attempted": len(specs), "n_kept": len(kept),
              "n_dropped": len(dropped), "dropped_by_reason": reasons, "dropped": dropped,
              "wall_seconds": round(wall, 1), "workers": args.workers,
              "solver": {"max_iters": C.MAX_ITERS, "conv_tol": C.CONV_TOL,
                         "conv_every": C.CONV_EVERY},
              "stratify_by": list(stratify), "re_bucket_edges": C.RE_BUCKET_EDGES,
              "split_sizes": {k: int(len(v)) for k, v in split.items()}, "split_counts": counts}
    dataset.write_dataset(args.out, inputs, targets, [m["cd_total"] for m in kept],
                          [m["cl_total"] for m in kept], split, header, kept)
    print(f"wrote {len(kept)} samples ({len(dropped)} dropped: {reasons}) to {args.out} "
          f"in {wall:.0f}s; split {header['split_sizes']}")


if __name__ == "__main__":
    main()
