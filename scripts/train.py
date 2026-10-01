"""Train the U-Net. (Owner: Abhay; needs torch). See ABHAY.md.

Usage:
    python scripts/train.py [--data DIR] [--out DIR] [--epochs 150] [--seed 0]
                            [--div-weight 0 1e-2 1 1e2 1e4] [--batch-size 16] [--lr 1e-3]

One arm is trained per --div-weight value, all with the same seed and budget.
Each arm logs to OUT/results/train_log_w<w>.csv and saves its best checkpoint
(by validation loss) to OUT/checkpoints/best_w<w>.pt. The arm with the lowest
validation relative L2 on velocity is copied to OUT/checkpoints/best.pt and
OUT/results/train_log.csv, and the comparison goes to OUT/results/ablation.json.

Only idx_train and idx_val are ever read. idx_test and idx_ood are not loaded.

Loss: per-channel MSE on normalised targets over fluid cells only (mask == 0),
averaged over u, v, p, plus w * mean(div^2), where div = du/dx + dv/dy by
central differences on physical u, v over interior fluid cells (cells next to
the obstacle or the walls are excluded).
"""

import argparse
import csv
import json
import os
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from windtunnel import contract as C  # noqa: E402
from windtunnel.model import build_model, param_count  # noqa: E402

TRAIN_SPLITS = ("idx_train", "idx_val")  # the only splits this script may read


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def load_split_data(data_dir):
    """Load train and val arrays plus norm.json. Never touches test/OOD."""
    with np.load(os.path.join(data_dir, C.DATASET_FILE)) as d:
        inputs, targets = d["inputs"], d["targets"]
        idx = {k: d[k] for k in TRAIN_SPLITS}
    with open(os.path.join(data_dir, C.NORM_FILE)) as f:
        norm = json.load(f)
    assert norm["computed_on"] == "idx_train", norm["computed_on"]
    assert tuple(norm["inputs"]["channels"]) == C.INPUT_CHANNELS
    assert tuple(norm["targets"]["channels"]) == C.TARGET_CHANNELS
    split = {k.removeprefix("idx_"): (torch.from_numpy(inputs[i]), torch.from_numpy(targets[i]))
             for k, i in idx.items()}
    return split, norm


def hardware_string():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "CPU only"
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"{gpu} / {cpu} ({os.cpu_count()} threads)"


class Normaliser:
    def __init__(self, norm, device):
        def t(v):
            return torch.tensor(v, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.in_mean, self.in_std = t(norm["inputs"]["mean"]), t(norm["inputs"]["std"])
        self.out_mean, self.out_std = t(norm["targets"]["mean"]), t(norm["targets"]["std"])

    def inputs(self, x):
        return (x - self.in_mean) / self.in_std

    def targets(self, y):
        return (y - self.out_mean) / self.out_std

    def denorm(self, y):
        return y * self.out_std + self.out_mean


def divergence_valid(mask):
    """Cells where the central-difference stencil is all fluid and away from the walls.

    mask: [N, 1, H, W] float (1 inside the obstacle). Dilating by a 3x3 window
    drops every cell next to the obstacle; rows 0, 1, NY-2, NY-1 (walls and the
    rows touching them) and the inlet/outlet columns are dropped too.
    """
    near_body = F.max_pool2d(mask, 3, stride=1, padding=1)
    valid = (1.0 - near_body)
    valid[:, :, :2] = 0
    valid[:, :, -2:] = 0
    valid[:, :, :, :1] = 0
    valid[:, :, :, -1:] = 0
    return valid


def divergence(u, v):
    """Central-difference du/dx + dv/dy on [N, H, W]; zero on the border cells."""
    div = torch.zeros_like(u)
    div[:, 1:-1, 1:-1] = ((u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / 2
                          + (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / 2)
    return div


def compute_losses(net, x_raw, y_raw, nrm, div_weight):
    """Returns (total, data, div) losses and the masked physical prediction."""
    mask = x_raw[:, C.IN_MASK:C.IN_MASK + 1]
    fluid = 1.0 - mask
    pred_n = net(nrm.inputs(x_raw))
    sq = (pred_n - nrm.targets(y_raw)) ** 2 * fluid
    n_fluid = fluid.sum().clamp_min(1.0)
    data = (sq.sum(dim=(0, 2, 3)) / n_fluid).mean()  # per-channel MSE, then channel mean
    phys = nrm.denorm(pred_n) * fluid
    valid = divergence_valid(mask)[:, 0]
    div = divergence(phys[:, C.OUT_U], phys[:, C.OUT_V])
    div_loss = (div**2 * valid).sum() / valid.sum().clamp_min(1.0)
    return data + div_weight * div_loss, data, div_loss, phys


def relative_l2_batch(pred, true, mask):
    """Per-sample relative L2 over fluid cells, physical units.

    pred, true: [N, 3, H, W]; mask: [N, H, W] (1 inside). Returns dict of [N]
    tensors for u, v, p and vel (stacked u, v), matching metrics.relative_l2.
    """
    fluid = (1.0 - mask)[:, None]
    err = ((pred - true) ** 2 * fluid).sum(dim=(2, 3))
    ref = (true**2 * fluid).sum(dim=(2, 3)).clamp_min(1e-30)
    out = {name: (err[:, k] / ref[:, k]).sqrt() for k, name in enumerate(C.TARGET_CHANNELS)}
    out["vel"] = ((err[:, C.OUT_U] + err[:, C.OUT_V]) / (ref[:, C.OUT_U] + ref[:, C.OUT_V])).sqrt()
    return out


@torch.no_grad()
def evaluate(net, data, nrm, div_weight, batch_size, device):
    net.eval()
    x_all, y_all = data
    tot = {"loss": 0.0, "data_loss": 0.0, "div_loss": 0.0}
    rel = {k: [] for k in ("u", "v", "p", "vel")}
    n = len(x_all)
    for s in range(0, n, batch_size):
        x, y = x_all[s:s + batch_size].to(device), y_all[s:s + batch_size].to(device)
        loss, data_l, div_l, phys = compute_losses(net, x, y, nrm, div_weight)
        for k, v in zip(tot, (loss, data_l, div_l)):
            tot[k] += v.item() * len(x)
        for k, v in relative_l2_batch(phys, y, x[:, C.IN_MASK]).items():
            rel[k].append(v.cpu())
    out = {k: v / n for k, v in tot.items()}
    out.update({f"rel_l2_{k}": torch.cat(v).mean().item() for k, v in rel.items()})
    return out


def weight_tag(w):
    return f"w{w:g}"


def train_arm(args, split, norm, div_weight, device, out_dir):
    seed_everything(args.seed)
    net = build_model({"base_channels": args.base_channels, "depth": args.depth}).to(device)
    nrm = Normaliser(norm, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    gen = torch.Generator().manual_seed(args.seed)  # deterministic batch order

    tag = weight_tag(div_weight)
    ckpt_path = os.path.join(out_dir, C.CHECKPOINT_DIR, f"best_{tag}.pt")
    log_path = os.path.join(out_dir, C.RESULTS_DIR, f"train_log_{tag}.csv")
    x_tr, y_tr = split["train"]
    best = {"val_loss": float("inf"), "epoch": 0}
    t0 = time.time()
    epochs_run = 0
    print(f"[{tag}] div_weight={div_weight:g} params={param_count(net):,} "
          f"train={len(x_tr)} val={len(split['val'][0])} device={device}", flush=True)
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds",
                      "train_data_loss", "train_div_loss", "val_rel_l2_vel"])
        for epoch in range(1, args.epochs + 1):
            net.train()
            lr = opt.param_groups[0]["lr"]
            perm = torch.randperm(len(x_tr), generator=gen)
            sums = np.zeros(3)
            for s in range(0, len(perm), args.batch_size):
                b = perm[s:s + args.batch_size]
                x, y = x_tr[b].to(device), y_tr[b].to(device)
                if args.flip_aug:
                    x, y = random_vflip(x, y, gen)
                loss, data_l, div_l, _ = compute_losses(net, x, y, nrm, div_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sums += np.array([loss.item(), data_l.item(), div_l.item()]) * len(b)
            sched.step()
            tr = sums / len(x_tr)
            val = evaluate(net, split["val"], nrm, div_weight, args.batch_size, device)
            secs = time.time() - t0
            log.writerow([epoch, f"{tr[0]:.6g}", f"{val['loss']:.6g}", f"{lr:.6g}", f"{secs:.1f}",
                          f"{tr[1]:.6g}", f"{tr[2]:.6g}", f"{val['rel_l2_vel']:.6g}"])
            f.flush()
            epochs_run = epoch
            if val["loss"] < best["val_loss"]:
                best = {"val_loss": val["loss"], "epoch": epoch, "val": val}
                torch.save({"state_dict": net.state_dict(), "config": net.config, "norm": norm,
                            "div_weight": div_weight, "epoch": epoch, "val": val,
                            "args": vars(args)}, ckpt_path)
            if epoch == 1 or epoch % args.print_every == 0:
                print(f"[{tag}] epoch {epoch:4d} train {tr[0]:.4g} (div {tr[2]:.3g}) "
                      f"val {val['loss']:.4g} relL2 u {val['rel_l2_u']:.3f} v {val['rel_l2_v']:.3f} "
                      f"p {val['rel_l2_p']:.3f} lr {lr:.2e} {secs:.0f}s", flush=True)
            if epoch - best["epoch"] >= args.patience:
                print(f"[{tag}] early stop at epoch {epoch} (best {best['epoch']})", flush=True)
                break
    seconds = time.time() - t0
    print(f"[{tag}] best epoch {best['epoch']} val {best['val']}", flush=True)
    return {"divergence_weight": div_weight, "best_epoch": best["epoch"], "epochs_run": epochs_run,
            "seconds": round(seconds, 1), "val": best["val"], "checkpoint": ckpt_path,
            "log": log_path, "param_count": param_count(net), "config": net.config}


def random_vflip(x, y, gen):
    """Mirror y -> -y on a random half of the batch, negating v. Only valid when the
    channel is symmetric about its centre line (it is: walls at rows 0 and NY-1)."""
    flip = torch.rand(len(x), generator=gen) < 0.5
    if flip.any():
        x, y = x.clone(), y.clone()
        x[flip] = x[flip].flip(-2)
        y[flip] = y[flip].flip(-2)
        y[flip, C.OUT_V] = -y[flip, C.OUT_V]
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=C.data_path(), help="dir with dataset.npz + norm.json")
    ap.add_argument("--out", default=None, help="dir for checkpoints/ and results/ (default: --data)")
    ap.add_argument("--div-weight", type=float, nargs="+", default=[0.0, 1e-2, 1.0, 1e2, 1e4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--flip-aug", action="store_true", help="vertical-flip augmentation (off)")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = args.out or args.data
    for sub in (C.CHECKPOINT_DIR, C.RESULTS_DIR):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    split, norm = load_split_data(args.data)
    device = torch.device(args.device)
    arms = [train_arm(args, split, norm, w, device, out_dir) for w in args.div_weight]

    # Choose on validation only: lowest relative L2 on velocity (stacked u, v).
    chosen = min(arms, key=lambda a: a["val"]["rel_l2_vel"])
    shutil.copyfile(chosen["checkpoint"], os.path.join(out_dir, C.CHECKPOINT_DIR, "best.pt"))
    shutil.copyfile(chosen["log"], os.path.join(out_dir, C.RESULTS_DIR, C.TRAIN_LOG_FILE))
    meta_path = os.path.join(args.data, C.META_FILE)
    with open(meta_path) as f:
        meta = json.load(f)
    summary = {
        "arms": arms,
        "chosen_divergence_weight": chosen["divergence_weight"],
        "selection": "lowest val rel L2 on velocity (stacked u, v), mean over val samples",
        "hardware": hardware_string(),
        "seeds": {"torch": args.seed, "numpy": args.seed, "python": args.seed},
        "dataset": {"dir": args.data, "n": len(meta["samples"]), "n_attempted": meta["n_attempted"],
                    "n_dropped": meta["n_dropped"], "seed": meta.get("seed"),
                    "fake": bool(meta.get("fake", False))},
        "args": vars(args),
        "argv": sys.argv,
    }
    with open(os.path.join(out_dir, C.RESULTS_DIR, "ablation.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("chosen:", weight_tag(chosen["divergence_weight"]),
          {k: round(v, 4) for k, v in chosen["val"].items()})


if __name__ == "__main__":
    main()
