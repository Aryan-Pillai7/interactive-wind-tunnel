"""Noise floor of the pressure-drag estimator. (Owner: Aryan)

The surface integral runs over a staircase boundary, so it depends on how a
shape happens to land on the grid. Measure it: solve the same circle at
several Re with its centre shifted by sub-cell offsets (0 and +-0.5 cell in x
and y), and report the spread of cd_p (and of the solver's cd_total) per Re.
A model's drag error below this spread is not meaningful.

Usage: python scripts/noise_floor.py [--d 12] [--re 10 25 40] [--workers 4]
Writes DATA_DIR/results/noise_floor.json.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windtunnel import contract as C  # noqa: E402
from windtunnel import geometry, lbm, metrics  # noqa: E402

OFFSETS = [(0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.5, 0.5)]


def run(job):
    d, re, off = job
    mask = geometry.make_mask("circle", d, offset=off)
    res = lbm.solve(mask, re)
    cd_p, cl_p = metrics.pressure_forces(res["p"], mask)
    return {"re": re, "offset": off, "d_measured": geometry.obstacle_height(mask),
            "n_cells": int(mask.sum()), "converged": res["converged"],
            "cd_p": cd_p, "cl_p": cl_p, "cd_total": res["cd_total"], "cl_total": res["cl_total"]}


def spread(v):
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "range": float(v.max() - v.min()),
            "rel_std": float(v.std() / abs(v.mean())), "rel_range": float((v.max() - v.min()) / abs(v.mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=12.0)
    ap.add_argument("--re", type=float, nargs="+", default=[10.0, 25.0, 40.0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(args.d, re, off) for re in args.re for off in OFFSETS]
    with mp.Pool(args.workers) as pool:
        runs = pool.map(run, jobs)
    per_re = {}
    for re in args.re:
        rs = [r for r in runs if r["re"] == re]
        per_re[str(re)] = {"cd_p": spread([r["cd_p"] for r in rs]),
                           "cd_total": spread([r["cd_total"] for r in rs]),
                           "cl_p_abs_max": float(max(abs(r["cl_p"]) for r in rs))}
        print(f"Re {re:>5}: cd_p {per_re[str(re)]['cd_p']['mean']:.3f} "
              f"rel std {per_re[str(re)]['cd_p']['rel_std']:.2%} "
              f"rel range {per_re[str(re)]['cd_p']['rel_range']:.2%} | "
              f"cd_total rel std {per_re[str(re)]['cd_total']['rel_std']:.2%}")
    floor = float(np.mean([v["cd_p"]["rel_std"] for v in per_re.values()]))
    out = {"method": "circle of nominal diameter d at each Re, centre shifted by sub-cell "
                     "offsets; spread of the pressure-drag estimate across offsets",
           "d": args.d, "offsets": OFFSETS, "per_re": per_re,
           "cd_p_noise_floor_rel_std": floor, "runs": runs}
    path = os.path.join(C.data_path(C.RESULTS_DIR), "noise_floor.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"noise floor (mean rel std of cd_p): {floor:.2%} -> {path}")


if __name__ == "__main__":
    main()
