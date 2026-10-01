"""Dataset assembly, splitting and loading, shared by the real and fake generators. (Owner: Aryan)"""

import json
import os

import numpy as np

from windtunnel import contract as C
from windtunnel.geometry import sdf


def build_inputs(mask, re):
    """Raw model input [3, NY, NX] float32: sdf, mask, re_norm."""
    mask = np.asarray(mask, bool)
    return np.stack([sdf(mask), mask.astype(np.float32),
                     np.full(mask.shape, C.re_to_norm(re), np.float32)])


def re_bucket(re):
    """Index of the RE_BUCKET_EDGES bucket containing re."""
    return int(np.searchsorted(C.RE_BUCKET_EDGES, re, side="right") - 1)


def split_indices(meta, rng, stratify_by=("kind", "re")):
    """Seeded train/val/test split of the in-distribution samples.

    stratify_by=("kind", "re"): SPLIT_FRACTIONS applied within each
    (kind, Re bucket); () gives one plain random split. OOD kinds go to
    idx_ood only. Returns ({train, val, test, ood}: int64 arrays, counts).
    """
    groups = {}
    for i, m in enumerate(meta):
        if m["kind"] in C.OOD_KINDS:
            continue
        key = tuple(m["kind"] if k == "kind" else re_bucket(m["re"]) for k in stratify_by)
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    sizes = np.array([len(groups[k]) for k in keys])
    # Largest-remainder allocation: totals match SPLIT_FRACTIONS even when
    # groups are small (plain per-group rounding sends small groups to train).
    alloc = {}
    for part in ("val", "test"):
        exact = sizes * C.SPLIT_FRACTIONS[part]
        n = np.floor(exact).astype(int)
        short = int(round(sizes.sum() * C.SPLIT_FRACTIONS[part])) - n.sum()
        order = np.lexsort((rng.random(len(keys)), -(exact - n)))
        n[order[:max(short, 0)]] += 1
        alloc[part] = n
    split = {"train": [], "val": [], "test": []}
    counts = {}
    for g, key in enumerate(keys):
        idx = np.array(groups[key])
        rng.shuffle(idx)
        n_val = int(alloc["val"][g])
        n_test = int(min(alloc["test"][g], len(idx) - n_val))
        parts = {"val": idx[:n_val], "test": idx[n_val:n_val + n_test],
                 "train": idx[n_val + n_test:]}
        for k, v in parts.items():
            split[k].extend(v.tolist())
        counts["|".join(map(str, key)) or "all"] = {k: len(v) for k, v in parts.items()}
    out = {k: np.sort(np.array(v, np.int64)) for k, v in split.items()}
    out["ood"] = np.array([i for i, m in enumerate(meta) if m["kind"] in C.OOD_KINDS], np.int64)
    return out, counts


def channel_stats(x):
    mean = x.mean(axis=(0, 2, 3), dtype=np.float64)
    std = np.maximum(x.std(axis=(0, 2, 3), dtype=np.float64), 1e-8)
    return {"mean": mean.tolist(), "std": std.tolist()}


def write_dataset(out_dir, inputs, targets, cd_total, cl_total, split, meta_header, samples):
    """Write dataset.npz, meta.json and norm.json (norm from idx_train only)."""
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, C.DATASET_FILE),
             inputs=np.asarray(inputs, np.float32), targets=np.asarray(targets, np.float32),
             cd_total=np.asarray(cd_total, np.float32), cl_total=np.asarray(cl_total, np.float32),
             idx_train=split["train"], idx_val=split["val"], idx_test=split["test"],
             idx_ood=split["ood"])
    with open(os.path.join(out_dir, C.META_FILE), "w") as f:
        json.dump(dict(meta_header, samples=samples), f, indent=1)
    tr = split["train"]
    with open(os.path.join(out_dir, C.NORM_FILE), "w") as f:
        json.dump({"computed_on": "idx_train",
                   "inputs": dict(channels=C.INPUT_CHANNELS, **channel_stats(inputs[tr])),
                   "targets": dict(channels=C.TARGET_CHANNELS, **channel_stats(targets[tr]))},
                  f, indent=1)


def load_dataset(data_dir=None):
    """Returns (arrays dict, meta dict, norm dict)."""
    data_dir = data_dir or C.DATA_DIR
    with np.load(os.path.join(data_dir, C.DATASET_FILE)) as z:
        arrays = {k: z[k] for k in z.files}
    with open(os.path.join(data_dir, C.META_FILE)) as f:
        meta = json.load(f)
    with open(os.path.join(data_dir, C.NORM_FILE)) as f:
        norm = json.load(f)
    return arrays, meta, norm
