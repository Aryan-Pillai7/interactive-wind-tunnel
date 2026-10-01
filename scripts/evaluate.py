"""Evaluate the surrogate and baselines on test and OOD. (Owner: Aryan)

Usage: python scripts/evaluate.py [--data DIR] [--model models/model.onnx]
           [--allow-dummy] [--solver-samples 3] [--out DIR]

This is the only code that touches idx_test and idx_ood. Writes
OUT/results.json (default DATA_DIR/results) and figures in OUT/figures/:
- relative L2 per channel (fluid cells), pressure drag/lift, stagnation
  pressure and wake vorticity errors, per model, on test and OOD;
- latency through onnxruntime (single shape, CPU) and the solver-vs-surrogate
  speed-up, both measured on this machine;
- near-duplicate report: each test sample's distance to its nearest training
  sample, and how the surrogate's error depends on it;
- the drag-estimator noise floor, if scripts/noise_floor.py has been run;
- the hypotheses from the plan, marked met / not met.
With --allow-dummy and no model file, the masked-uniform-flow dummy stands in
for the surrogate and everything is labelled "dummy".
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import spearmanr  # noqa: E402

from windtunnel import baselines, geometry, lbm, metrics, viz  # noqa: E402
from windtunnel import contract as C  # noqa: E402
from windtunnel.dataset import load_dataset  # noqa: E402
from windtunnel.surrogate import load_surrogate  # noqa: E402

TARGETS = {"vel_rel_l2": 0.10, "cd_p_rel_err": 0.10, "latency_ms": 50.0, "speedup": 100.0}


def summarise(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def rel_err(a, b, floor=1e-6):
    """|a - b| / |b|, NaN when the reference is ~0 (e.g. lift of a symmetric body)."""
    return abs(a - b) / abs(b) if abs(b) > floor else float("nan")


def sample_errors(pred, true, mask, true_m):
    rel = metrics.relative_l2(pred, true, mask)
    pm = metrics.field_metrics(pred, mask)
    return {
        "rel_l2_u": rel["u"], "rel_l2_v": rel["v"], "rel_l2_p": rel["p"], "rel_l2_vel": rel["vel"],
        "cd_p_rel_err": rel_err(pm["cd_p"], true_m["cd_p"]),
        "cl_p_abs_err": abs(pm["cl_p"] - true_m["cl_p"]),
        "stagnation_p_abs_err": abs(pm["stagnation_p"] - true_m["stagnation_p"]),
        "wake_vorticity_rel_err": rel_err(pm["wake_vorticity"], true_m["wake_vorticity"]),
        "div_ms": pm["div_ms"],
    }


def evaluate_split(name, idx, arrays, samples, models, surrogate_preds):
    """Per-model summaries, per-sample surrogate errors, and per-model velocity errors."""
    per_model = {m: [] for m in models}
    per_sample = []
    for k, i in enumerate(idx):
        x, true = arrays["inputs"][i], arrays["targets"][i]
        mask = x[C.IN_MASK] > 0.5
        kind = samples[i]["kind"]
        true_m = metrics.field_metrics(true, mask)
        for mname, model in models.items():
            if mname == "surrogate":
                pred = surrogate_preds[k]
            elif mname == "potential_flow":
                pred = model.predict(x, kind)
            else:
                pred = model.predict(x)
            if pred is None:
                continue
            err = sample_errors(pred, true, mask, true_m)
            per_model[mname].append(err)
            if mname == "surrogate":
                per_sample.append(dict(err, index=int(i), kind=kind, re=samples[i]["re"],
                                       cd_p_true=true_m["cd_p"],
                                       cd_total=float(arrays["cd_total"][i])))
    summary = {}
    for mname, errs in per_model.items():
        keys = errs[0].keys() if errs else []
        summary[mname] = {k: summarise([e[k] for e in errs]) for k in keys}
        summary[mname]["n_samples"] = len(errs)
    true_div = [metrics.mean_sq_divergence(*arrays["targets"][i][:2], arrays["inputs"][i][C.IN_MASK] > 0.5)
                for i in idx]
    summary["ground_truth_div_ms"] = summarise(true_div)
    print(f"\n== {name} ({len(idx)} samples) ==")
    print(f"{'model':<20}{'vel rel L2':>12}{'p rel L2':>11}{'Cd_p err':>11}{'Cl_p abs':>11}")
    for mname in models:
        s = summary[mname]
        if s["n_samples"]:
            row = [s[k].get("mean", float("nan")) for k in
                   ("rel_l2_vel", "rel_l2_p", "cd_p_rel_err", "cl_p_abs_err")]
            print(f"{mname:<20}{row[0]:>12.4f}{row[1]:>11.4f}{row[2]:>11.4f}{row[3]:>11.4f}"
                  f"   (n={s['n_samples']})")
    vel = {m: np.array([e["rel_l2_vel"] for e in errs]) for m, errs in per_model.items() if errs}
    return summary, per_sample, vel


def measure_latency(model, x, reps):
    for _ in range(5):
        model.predict(x)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "reps": reps}


def measure_solver(samples, idx, n):
    out = []
    for i in idx[:n]:
        s = samples[i]
        mask = geometry.make_mask(s["kind"], **s["params"])
        res = lbm.solve(mask, s["re"], d=s["d"])
        out.append({"index": int(i), "kind": s["kind"], "re": s["re"],
                    "seconds": res["seconds"], "iterations": res["iterations"]})
        print(f"solver on test sample {i} ({s['kind']}, Re {s['re']:.1f}): "
              f"{res['seconds']:.1f}s, {res['iterations']} iterations")
    return out


def near_duplicates(nn, arrays, idx_train, idx_val, idx_test, per_sample):
    _, d_test = nn.query(arrays["inputs"][idx_test])
    _, d_val = nn.query(arrays["inputs"][idx_val]) if len(idx_val) else (None, np.array([]))
    _, d_loo = nn.query(arrays["inputs"][idx_train], exclude_self=True)
    ref = float(np.median(d_loo))
    errs = np.array([p["rel_l2_vel"] for p in per_sample])
    ok = np.isfinite(errs)
    rho = spearmanr(d_test[ok], errs[ok]).statistic if ok.sum() > 2 else float("nan")
    for p, d in zip(per_sample, d_test):
        p["nn_distance"] = float(d)
    return {
        "distance": "RMS over cells and channels of norm.json-normalised inputs",
        "test_to_train": summarise(d_test), "val_to_train": summarise(d_val),
        "train_leave_one_out": summarise(d_loo),
        "frac_test_closer_than_median_train_loo": float(np.mean(d_test < ref)),
        "spearman_surrogate_vel_err_vs_distance": float(rho),
        "note": "A test set much closer to train than train is to itself would inflate "
                "in-distribution scores; a positive Spearman means errors grow away from train.",
    }


def save_figures(out_dir, arrays, idx_test, idx_ood, test_preds, ood_preds, per_test, per_ood,
                 per_model_errors, label):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    order = np.argsort([p["rel_l2_vel"] for p in per_test])
    picks = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
    for tag, k in picks.items():
        i = idx_test[k]
        mask = arrays["inputs"][i][C.IN_MASK] > 0.5
        p = per_test[k]
        fig = viz.plot_comparison(test_preds[k], arrays["targets"][i], mask,
                                  f"{label} - test {tag}: {p['kind']}, Re {p['re']:.1f}, "
                                  f"vel rel L2 {p['rel_l2_vel']:.3f}")
        path = os.path.join(fig_dir, f"test_{tag}.png")
        fig.savefig(path, dpi=110)
        written.append(path)
    if len(idx_ood):
        k = int(np.argmax([p["rel_l2_vel"] for p in per_ood]))
        i = idx_ood[k]
        mask = arrays["inputs"][i][C.IN_MASK] > 0.5
        fig = viz.plot_comparison(ood_preds[k], arrays["targets"][i], mask,
                                  f"{label} - OOD worst: triangle, Re {per_ood[k]['re']:.1f}, "
                                  f"vel rel L2 {per_ood[k]['rel_l2_vel']:.3f}")
        path = os.path.join(fig_dir, "ood_worst.png")
        fig.savefig(path, dpi=110)
        written.append(path)
    fig = viz.plot_error_hist(per_model_errors, "velocity relative L2 on test", "relative L2")
    path = os.path.join(fig_dir, "test_vel_error_hist.png")
    fig.savefig(path, dpi=110)
    written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.DATA_DIR)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--allow-dummy", action="store_true")
    ap.add_argument("--solver-samples", type=int, default=3)
    ap.add_argument("--latency-reps", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.data, C.RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    arrays, meta, norm = load_dataset(args.data)
    samples = meta["samples"]
    idx_tr, idx_va = arrays["idx_train"], arrays["idx_val"]
    idx_te, idx_ood = arrays["idx_test"], arrays["idx_ood"]

    surrogate = load_surrogate(args.model, allow_dummy=args.allow_dummy)
    label = "dummy" if surrogate.is_dummy else "surrogate"
    print(f"model: {args.model} ({label}); dataset: {args.data} "
          f"(train {len(idx_tr)}, val {len(idx_va)}, test {len(idx_te)}, ood {len(idx_ood)})")

    tr_x, tr_y = arrays["inputs"][idx_tr], arrays["targets"][idx_tr]
    nn = baselines.NearestNeighbourBaseline(norm).fit(tr_x, tr_y)
    models = {"surrogate": surrogate,
              "mean_field": baselines.MeanFieldBaseline().fit(tr_x, tr_y),
              "nearest_neighbour": nn,
              "potential_flow": baselines.PotentialFlowBaseline()}

    test_preds = surrogate.predict_batch(arrays["inputs"][idx_te])
    ood_preds = surrogate.predict_batch(arrays["inputs"][idx_ood]) if len(idx_ood) else []
    test_summary, per_test, test_vel = evaluate_split("test", idx_te, arrays, samples, models,
                                                      test_preds)
    ood_summary, per_ood, _ = evaluate_split("ood", idx_ood, arrays, samples, models, ood_preds) \
        if len(idx_ood) else ({}, [], {})

    latency = measure_latency(surrogate, arrays["inputs"][idx_te[0]], args.latency_reps)
    solver_runs = measure_solver(samples, idx_te, args.solver_samples)
    solver_s = float(np.median([r["seconds"] for r in solver_runs])) if solver_runs else None
    speedup = solver_s * 1000 / latency["median_ms"] if solver_s else None
    print(f"\nlatency {latency['median_ms']:.2f} ms (median); "
          + (f"solver {solver_s:.1f} s -> speed-up {speedup:,.0f}x" if speedup else
             "solver not timed"))

    dup = near_duplicates(nn, arrays, idx_tr, idx_va, idx_te, per_test)
    print(f"near-duplicates: {dup['frac_test_closer_than_median_train_loo']:.0%} of test samples "
          f"are closer to train than the median train sample is to its neighbour; "
          f"Spearman(err, distance) = {dup['spearman_surrogate_vel_err_vs_distance']:.2f}")

    noise_path = os.path.join(out_dir, "noise_floor.json")
    noise = json.load(open(noise_path)) if os.path.exists(noise_path) else None

    s = test_summary["surrogate"]
    measured = {"vel_rel_l2": s["rel_l2_vel"].get("mean"),
                "cd_p_rel_err": s["cd_p_rel_err"].get("mean"),
                "latency_ms": latency["median_ms"], "speedup": speedup}
    hypotheses = {}
    for k, target in TARGETS.items():
        v = measured[k]
        ok = None if v is None else (v >= target if k == "speedup" else v < target)
        hypotheses[k] = {"target": ("<" if k != "speedup" else ">=") + str(target),
                         "measured": v, "met": ok}

    test_vel = {(label if m == "surrogate" else m): v for m, v in test_vel.items()
                if m != "potential_flow"}
    figures = save_figures(out_dir, arrays, idx_te, idx_ood, test_preds, ood_preds, per_test,
                           per_ood, test_vel, label)

    results = {
        "model": {"path": args.model, "label": label,
                  "card": json.load(open(C.MODEL_CARD_PATH)) if os.path.exists(C.MODEL_CARD_PATH)
                  and not surrogate.is_dummy else None},
        "dataset": {"dir": args.data, "fake": meta.get("fake"), "seed": meta.get("seed"),
                    "n_kept": len(samples), "n_dropped": meta.get("n_dropped"),
                    "split_sizes": {k: int(len(arrays[f"idx_{k}"])) for k in ("train", "val", "test", "ood")}},
        "how_measured": {
            "rel_l2": "per channel over fluid cells, ||pred-true||/||true||; vel stacks u and v",
            "cd_p": "pressure-only surface integral over the staircase boundary, same estimator "
                    "on ground truth and prediction; relative error",
            "cl_p": "absolute error (true lift is often ~0, so relative error is meaningless)",
            "stagnation_p": "max p on fluid cells adjacent to the obstacle; absolute error",
            "wake_vorticity": "mean |w| D/U0 over 2D downstream window; relative error",
            "latency": f"onnxruntime CPU, single shape, median of {args.latency_reps} after 5 warm-up",
            "speedup": "median solver wall time on the first test samples / median latency, same machine",
            "machine": {"platform": platform.platform(), "cpus": os.cpu_count()},
        },
        "hypotheses": hypotheses,
        "test": test_summary, "ood": ood_summary,
        "latency": latency, "solver_runs": solver_runs, "speedup": speedup,
        "near_duplicates": dup, "noise_floor": noise,
        "per_sample": {"test": per_test, "ood": per_ood},
        "figures": [os.path.relpath(p, out_dir) for p in figures],
    }
    with open(os.path.join(out_dir, C.RESULTS_FILE), "w") as f:
        json.dump(results, f, indent=1)
    print("\nhypotheses:", {k: v["met"] for k, v in hypotheses.items()})
    print(f"wrote {os.path.join(out_dir, C.RESULTS_FILE)} and {len(figures)} figures")


if __name__ == "__main__":
    main()
