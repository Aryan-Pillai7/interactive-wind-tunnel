"""Export the best checkpoint to models/model.onnx and write model_card.json.
(Owner: Abhay; needs torch + onnx). See ABHAY.md for the exact contract.

Usage:
    python scripts/export_onnx.py [--data DIR] [--out DIR] [--ckpt PATH]
                                  [--onnx models/model.onnx] [--card models/model_card.json]

Reads OUT/checkpoints/best.pt and OUT/results/ablation.json written by
scripts/train.py (OUT defaults to --data). The torch-vs-onnxruntime check runs
on the validation set only (idx_val), with batch size 1 and with a batch > 1.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from windtunnel import contract as C  # noqa: E402
from windtunnel.model import Exported, build_model, param_count  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_exported(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = build_model(ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    n = ckpt["norm"]
    model = Exported(net, n["inputs"]["mean"], n["inputs"]["std"],
                     n["targets"]["mean"], n["targets"]["std"]).eval()
    return model, ckpt


def export(model, path):
    dummy = torch.zeros(2, C.N_IN, C.NY, C.NX, dtype=torch.float32)
    dummy[:, C.IN_RE] = 0.5
    torch.onnx.export(model, (dummy,), path, opset_version=C.ONNX_OPSET,
                      input_names=[C.ONNX_INPUT], output_names=[C.ONNX_OUTPUT],
                      dynamic_axes={C.ONNX_INPUT: {0: C.ONNX_BATCH_AXIS},
                                    C.ONNX_OUTPUT: {0: C.ONNX_BATCH_AXIS}},
                      dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return m


def check_io(m):
    """The graph must match the dummy reference: names, opset, dynamic batch axis."""
    assert [i.name for i in m.graph.input] == [C.ONNX_INPUT], m.graph.input
    assert [o.name for o in m.graph.output] == [C.ONNX_OUTPUT], m.graph.output
    assert any(o.domain in ("", "ai.onnx") and o.version == C.ONNX_OPSET for o in m.opset_import)
    for vi, ch in ((m.graph.input[0], C.N_IN), (m.graph.output[0], C.N_OUT)):
        dims = vi.type.tensor_type.shape.dim
        assert dims[0].dim_param == C.ONNX_BATCH_AXIS, dims[0]
        assert [d.dim_value for d in dims[1:]] == [ch, C.NY, C.NX], dims


def torch_vs_onnx(model, path, x_val):
    """Max |torch - onnxruntime| over the whole val set, at batch 1 and batch > 1."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(x_val)).numpy()
    diffs = {}
    for bs in (1, min(8, len(x_val))):
        out = np.concatenate([sess.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: x_val[s:s + bs]})[0]
                              for s in range(0, len(x_val), bs)])
        assert out.shape == ref.shape and out.dtype == np.float32, (out.shape, out.dtype)
        mask = x_val[:, C.IN_MASK:C.IN_MASK + 1] > 0.5
        assert np.all(out * mask == 0), "fields must be exactly 0 inside the obstacle"
        diffs[f"batch_{bs}"] = float(np.abs(out - ref).max())
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.data_path())
    ap.add_argument("--out", default=None, help="training output dir (default: --data)")
    ap.add_argument("--ckpt", default=None, help="default: OUT/checkpoints/best.pt")
    ap.add_argument("--onnx", default=os.path.join(REPO, C.MODEL_PATH))
    ap.add_argument("--card", default=os.path.join(REPO, C.MODEL_CARD_PATH))
    ap.add_argument("--hardware", default=None, help="override the detected hardware string")
    args = ap.parse_args()
    out_dir = args.out or args.data
    ckpt_path = args.ckpt or os.path.join(out_dir, C.CHECKPOINT_DIR, "best.pt")
    with open(os.path.join(out_dir, C.RESULTS_DIR, "ablation.json")) as f:
        ablation = json.load(f)

    model, ckpt = load_exported(ckpt_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx)), exist_ok=True)
    m = export(model, args.onnx)
    check_io(m)

    with np.load(os.path.join(args.data, C.DATASET_FILE)) as d:
        x_val = d["inputs"][d["idx_val"]]  # validation only
    diffs = torch_vs_onnx(model, args.onnx, x_val)
    max_diff = max(diffs.values())
    size_mb = os.path.getsize(args.onnx) / 2**20
    print(f"torch vs onnx max abs diff on val ({len(x_val)} samples): {diffs}; size {size_mb:.2f} MB")
    assert max_diff < C.TORCH_ONNX_MAX_ABS_DIFF, f"{max_diff} >= {C.TORCH_ONNX_MAX_ABS_DIFF}"
    if size_mb >= C.MODEL_MAX_COMMIT_MB:
        print(f"WARNING: {size_mb:.1f} MB >= {C.MODEL_MAX_COMMIT_MB} MB, do not commit model.onnx")

    arms = ablation["arms"]
    chosen_w = ckpt["div_weight"]
    keys = ("loss", "rel_l2_u", "rel_l2_v", "rel_l2_p", "rel_l2_vel")

    def arm_entry(a):
        return {"divergence_weight": a["divergence_weight"], "best_epoch": a["best_epoch"],
                "epochs_run": a["epochs_run"], "seconds": a["seconds"],
                "val": {k: a["val"][k] for k in keys}}

    no_div = next(a for a in arms if a["divergence_weight"] == 0)
    with_div = [a for a in arms if a["divergence_weight"] > 0]
    best_div = min(with_div, key=lambda a: a["val"]["rel_l2_vel"]) if with_div else None
    chosen_arm = next(a for a in arms if a["divergence_weight"] == chosen_w)
    if best_div is None:
        reason = "only the w = 0 arm was trained"
    else:
        a, b = no_div["val"]["rel_l2_vel"], best_div["val"]["rel_l2_vel"]
        reason = (f"lower val rel L2 on velocity (stacked u, v): "
                  f"{min(a, b):.4f} vs {max(a, b):.4f}")
    ds = ablation["dataset"]
    a = ablation["args"]
    data_flag = "" if os.path.normpath(args.data) == os.path.normpath(C.data_path()) else \
        f" --data {args.data}"
    out_flag = "" if out_dir == args.data else f" --out {out_dir}"
    gen_cmd = (f"python scripts/make_fake_dataset.py --seed {ds['seed']}" if ds["fake"] else
               f"docker compose run --rm gen --n <N> --workers <cores> --seed {ds['seed']}")
    card = {
        "param_count": param_count(model.net),
        "config": {**ckpt["config"], "architecture": "U-Net, DoubleConv(3x3 conv, GroupNorm, GELU) "
                   "x2 per level, maxpool down, 2x2 transposed conv up, concat skips, 1x1 head",
                   "grid": [C.NY, C.NX], "input_channels": list(C.INPUT_CHANNELS),
                   "target_channels": list(C.TARGET_CHANNELS)},
        "loss": {"type": "per-channel normalised MSE + w * mean(div^2)",
                 "divergence_weight": chosen_w,
                 "data_term": "MSE on norm.json-normalised targets over fluid cells only "
                              "(mask == 0), per channel, then mean over u, v, p",
                 "divergence_term": "central differences on physical (denormalised) u, v; mean "
                                    "over interior fluid cells, excluding cells next to the "
                                    "obstacle (3x3 dilation), rows 0-1 and NY-2..NY-1, and the "
                                    "first/last column",
                 "augmentation": "vertical flip (v negated)" if a["flip_aug"] else "none"},
        "ablation": {
            "no_divergence": arm_entry(no_div),
            "with_divergence": arm_entry(best_div) if best_div else None,
            "sweep": [arm_entry(x) for x in arms],
            "chosen": "no_divergence" if chosen_w == 0 else "with_divergence",
            "reason": reason,
            "selection": ablation["selection"] + "; test and OOD never used",
        },
        "training": {"epochs_run": chosen_arm["epochs_run"], "best_epoch": chosen_arm["best_epoch"],
                     "seconds": chosen_arm["seconds"],
                     "seconds_all_arms": round(sum(x["seconds"] for x in arms), 1),
                     "hardware": args.hardware or ablation["hardware"],
                     "seeds": ablation["seeds"],
                     "optimizer": f"Adam lr {a['lr']}, cosine to 0 over {a['epochs']} epochs",
                     "batch_size": a["batch_size"], "max_epochs": a["epochs"],
                     "early_stopping": f"val loss, patience {a['patience']}",
                     "dataset": ds},
        "validation": {k: chosen_arm["val"][k] for k in keys} | {"n": int(len(x_val))},
        "onnx": {"opset": C.ONNX_OPSET, "torch_vs_onnx_max_abs_diff_val": max_diff,
                 "torch_vs_onnx_by_batch_size": diffs, "size_mb": round(size_mb, 3),
                 "input": C.ONNX_INPUT, "output": C.ONNX_OUTPUT,
                 "io": "raw inputs [batch,3,64,128] -> physical u, v, p [batch,3,64,128], "
                       "multiplied by (1 - mask)"},
        "reproduce": [
            gen_cmd,
            "python scripts/train.py" + data_flag + out_flag
            + f" --seed {a['seed']} --epochs {a['epochs']} --patience {a['patience']}"
            + f" --batch-size {a['batch_size']} --lr {a['lr']}"
            + " --div-weight " + " ".join(f"{x['divergence_weight']:g}" for x in arms)
            + (" --flip-aug" if a["flip_aug"] else ""),
            "python scripts/export_onnx.py" + data_flag + out_flag,
        ],
    }
    with open(args.card, "w") as f:
        json.dump(card, f, indent=1)
    print(f"wrote {args.onnx} and {args.card}")


if __name__ == "__main__":
    main()
