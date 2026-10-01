"""Train the tiny MLP surrogate (v0.1), export it to ONNX and write the model card.

Usage:
    python scripts/train_mlp.py [--data DATA_DIR/tiny_dataset.npz] [--epochs 5] [--seed 0]

The npz needs `inputs` and `targets` [N, 3, 64, 128]. If it has idx_train and
idx_val they are used; otherwise the samples are split 80/20 with a seeded
shuffle (40 / 10 for 50 samples). Normalisation stats come from norm.json next
to the npz if present, else from the train split. Writes
DATA_DIR/checkpoints/best_model.pt, DATA_DIR/results/train_log_mlp.csv,
models/model.onnx and models/model_card.json.
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from windtunnel import contract as C  # noqa: E402
from windtunnel.model import MLP, Exported, param_count  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ONNX_TOL = 1e-3  # v0.1 brief; the contract value C.TORCH_ONNX_MAX_ABS_DIFF is also reported


def load(path, seed):
    with np.load(path) as d:
        x, y = d["inputs"].astype(np.float32), d["targets"].astype(np.float32)
        if "idx_train" in d.files and "idx_val" in d.files:
            tr, va = d["idx_train"], d["idx_val"]
        else:
            perm = np.random.default_rng(seed).permutation(len(x))
            n_val = int(round(0.2 * len(x)))
            tr, va = np.sort(perm[n_val:]), np.sort(perm[:n_val])
    norm_path = os.path.join(os.path.dirname(path), C.NORM_FILE)
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            n = json.load(f)
        stats = [n["inputs"]["mean"], n["inputs"]["std"], n["targets"]["mean"], n["targets"]["std"]]
        source = "norm.json next to the npz"
    else:
        def ms(a):
            return a.mean(axis=(0, 2, 3)).tolist(), np.maximum(a.std(axis=(0, 2, 3)), 1e-8).tolist()
        stats = [*ms(x[tr]), *ms(y[tr])]
        source = "computed on the train split"
    return x, y, tr, va, stats, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.data_path("tiny_dataset.npz"))
    ap.add_argument("--out", default=C.DATA_DIR, help="dir for checkpoints/ and results/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onnx", default=os.path.join(REPO, C.MODEL_PATH))
    ap.add_argument("--card", default=os.path.join(REPO, C.MODEL_CARD_PATH))
    ap.add_argument("--hardware", default=f"{platform.processor()} ({os.cpu_count()} threads)")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x, y, tr, va, stats, norm_source = load(args.data, args.seed)
    model = Exported(MLP(), *stats)
    net = model.net
    print(f"MLP params {param_count(net):,}; train {len(tr)} val {len(va)}; data {args.data}")

    # Loss: MSE on normalised (u, v, p), all cells.
    def loss_fn(xb, yb):
        pred = net((xb - model.in_mean) / model.in_std)
        return ((pred - (yb - model.out_mean) / model.out_std) ** 2).mean()

    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    ckpt_dir, res_dir = os.path.join(args.out, C.CHECKPOINT_DIR), os.path.join(args.out, C.RESULTS_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    log_path = os.path.join(res_dir, "train_log_mlp.csv")
    best_val, best_epoch, t0 = float("inf"), 0, time.time()
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "seconds"])
        for epoch in range(1, args.epochs + 1):
            net.train()
            perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=gen)]
            total = 0.0
            for s in range(0, len(perm), args.batch_size):
                b = perm[s:s + args.batch_size]
                loss = loss_fn(xt[b], yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            net.eval()
            with torch.no_grad():
                val = loss_fn(xt[va], yt[va]).item()
            secs = time.time() - t0
            log.writerow([epoch, f"{total / len(tr):.6g}", f"{val:.6g}", f"{secs:.2f}"])
            print(f"epoch {epoch} train {total / len(tr):.4f} val {val:.4f} {secs:.2f}s")
            if val < best_val:
                best_val, best_epoch = val, epoch
                torch.save({"state_dict": net.state_dict(), "stats": stats, "epoch": epoch,
                            "val_loss": val}, ckpt_path)
    train_seconds = time.time() - t0

    # Export the best checkpoint with normalisation + masking baked in.
    net.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx)), exist_ok=True)
    torch.onnx.export(model, (xt[:1],), args.onnx, opset_version=C.ONNX_OPSET,
                      input_names=[C.ONNX_INPUT], output_names=[C.ONNX_OUTPUT],
                      dynamic_axes={C.ONNX_INPUT: {0: C.ONNX_BATCH_AXIS},
                                    C.ONNX_OUTPUT: {0: C.ONNX_BATCH_AXIS}}, dynamo=False)

    # Check: one sample, a batch (dynamic axis), whole val set, masking, latency.
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    one = x[va[:1]]
    with torch.no_grad():
        ref_one, ref_val = model(torch.from_numpy(one)).numpy(), model(xt[va]).numpy()
    out_one = sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: one})[0]
    out_val = sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: x[va]})[0]
    diff_one = float(np.abs(out_one - ref_one).max())
    diff_val = float(np.abs(out_val - ref_val).max())
    assert out_val.shape == (len(va), C.N_OUT, C.NY, C.NX), out_val.shape
    assert np.all(out_val[np.broadcast_to(x[va][:, 1:2] > 0.5, out_val.shape)] == 0), "not masked"
    assert max(diff_one, diff_val) < ONNX_TOL, (diff_one, diff_val)
    for _ in range(20):
        sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: one})
    t = time.perf_counter()
    for _ in range(200):
        sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: one})
    latency_ms = (time.perf_counter() - t) / 200 * 1000
    size_mb = os.path.getsize(args.onnx) / 2**20
    print(f"onnx diff one {diff_one:.2e} val {diff_val:.2e}; latency {latency_ms:.3f} ms; {size_mb:.1f} MB")

    meta_path = os.path.join(os.path.dirname(args.data), C.META_FILE)
    fake = False
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            fake = bool(json.load(f).get("fake", False))
    card = {
        "version": "v0.1",
        "model": "MLP: flatten 24576 -> 64 -> 64 -> 24576, ReLU",
        "param_count": param_count(net),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "training_seconds": round(train_seconds, 2),
        "training": {"epochs": args.epochs, "optimizer": f"Adam lr {args.lr}, no schedule",
                     "batch_size": args.batch_size, "loss": "MSE on normalised u, v, p (all cells)",
                     "seed": args.seed, "n_train": int(len(tr)), "n_val": int(len(va)),
                     "normalisation": norm_source,
                     "data": os.path.basename(args.data) + (" (FAKE potential-flow data)" if fake else ""),
                     "hardware": f"{args.hardware}, CPU only, torch {torch.__version__}"},
        "onnx": {"opset": C.ONNX_OPSET, "input": C.ONNX_INPUT, "output": C.ONNX_OUTPUT,
                 "dynamic_batch": True, "size_mb": round(size_mb, 2),
                 "torch_vs_onnx_max_abs_diff_one_sample": diff_one,
                 "torch_vs_onnx_max_abs_diff_val": diff_val,
                 "passes_contract_1e-4": max(diff_one, diff_val) < C.TORCH_ONNX_MAX_ABS_DIFF,
                 "inference_ms_per_sample_cpu_onnxruntime": round(latency_ms, 3),
                 "io": "raw inputs -> physical u, v, p, multiplied by (1 - mask)"},
        "reproduce": [f"python scripts/train_mlp.py --data <DATA_DIR>/"
                      f"{os.path.relpath(args.data, C.DATA_DIR).replace(os.sep, '/')} "
                      f"--epochs {args.epochs} --seed {args.seed}"],
    }
    with open(args.card, "w") as f:
        json.dump(card, f, indent=1)
    print(f"wrote {args.onnx}, {args.card}, {ckpt_path}, {log_path}")


if __name__ == "__main__":
    main()
