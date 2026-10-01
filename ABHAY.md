# ABHAY.md: your half of interactive-wind-tunnel

This file should let you work without asking Aryan anything. If something here is unclear or
looks wrong, raise it in your PR. Don't silently work around it.

## 1. What the project is, and why

A **neural surrogate for steady 2D fluid flow**, presented as an interactive virtual wind tunnel.
A user picks a 2D shape (circle, ellipse, rotated rectangle), sets its size, angle and Reynolds number
with sliders, and a trained network predicts the full flow field around it (velocity `u`, `v` and
pressure `p`) in milliseconds. The app draws streamlines and velocity and pressure maps. It also reports
quantities computed from the predicted field: pressure drag and lift coefficients, peak stagnation
pressure, wake vorticity and inference time. It can run the real solver on demand to show
"surrogate vs ground truth" side by side.

What we are claiming, stated honestly:

- **The "wind tunnel" is a simulated channel**, 128 x 64 lattice cells with solid top and bottom walls.
  It is not a physical tunnel and not open air.
- **The ground truth is our own solver**: a D2Q9 lattice-Boltzmann (LBM) code in numpy that Aryan writes.
  The surrogate learns to imitate *that solver*, including its discretisation errors. It is only as good
  as the solver.
- **2D only.** Real flow past bodies is 3D. 2D is a deliberate simplification so the project fits on our hardware.
- **Re 10-40 only.** Above Re ~47, flow behind a bluff body starts shedding vortices (the von Kármán street)
  and never becomes steady, so one shape would no longer have one answer. We only learn the steady regime.
- **Blockage.** Obstacles are 10-16 cells tall in a 62-cell channel (up to ~26% blockage). Walls affect
  the flow, so drag values are channel values, not free-stream values.
- NACA airfoils are **deferred**. They are too thin to resolve cleanly on this grid. They may come back later.

It is a deep learning project: supervised learning of geometry + Re -> flow field, with a soft physics
prior (a divergence penalty) in the loss.

## 2. Goals and targets

These are **hypotheses we test, not promises**. We report whatever we measure.

| Metric (in-distribution test set) | Target |
|---|---|
| Relative L2 error on velocity (u, v), fluid cells | < 10% |
| Pressure-drag coefficient error (same surface estimator on GT and prediction) | < 10% |
| Inference per shape, CPU, onnxruntime | < 50 ms |
| Speed-up vs the LBM solver per shape (Aryan's machine) | >= 100x |
| OOD set (triangles) | reported, **no target** |

**Rules you must follow:**

- **The test set (`idx_test`) and the OOD set (`idx_ood`) are never used for anything during your work.**
  Not for the divergence weight, not for picking the epoch or checkpoint, not for architecture choices,
  not for learning-rate tuning, not even "just to look". Load them nowhere in `train.py`. Aryan's `evaluate.py`
  is the only code that touches them.
- All choices are made on `idx_val` only.
- Never change targets, normalisation, metrics or splits to make numbers look better. If results are bad, report them.

## 3. Who owns what

**You own (create and edit freely):**

- `windtunnel/model.py`: the U-Net (currently an empty stub)
- `scripts/train.py`: training (stub)
- `scripts/export_onnx.py`: ONNX export + `model_card.json` (stub)
- `models/model.onnx`, `models/model_card.json`: your deliverables
- The `train` stage in `Dockerfile` and the `train`/`export` services in `compose.yaml` (only if they need fixing; say so in the PR)
- `requirements-train.txt` (don't add packages without asking Aryan)
- Optional: `tests/test_model.py` and similar for your own code

**You must not edit** (raise an issue or PR comment instead):

- `windtunnel/contract.py`: the shared contract
- `windtunnel/geometry.py`, `lbm.py`, `metrics.py`, `baselines.py`, `surrogate.py`, `viz.py`, `dummy_model.py`
- `scripts/generate.py`, `scripts/make_fake_dataset.py`, `scripts/evaluate.py`, `app/`
- The `base` stage of `Dockerfile`, `requirements-base.txt`, the base services in `compose.yaml`

**Conflict rule:** if your code and Aryan's disagree (shapes, units, sign conventions, normalisation,
metric definitions), **Aryan's implementation is the reference**. Adapt your code to it, and if you think
his is wrong, say so in your PR with the evidence.

## 4. The contract (exact)

All constants live in `windtunnel/contract.py`. **Import them; never hard-code them.**

```python
from windtunnel import contract as C
C.NX, C.NY                  # 128, 64
C.U0                        # 0.05 (lattice units)
C.RE_MIN, C.RE_MAX          # 10.0, 40.0
C.INPUT_CHANNELS            # ("sdf", "mask", "re_norm")
C.TARGET_CHANNELS           # ("u", "v", "p")
C.ONNX_INPUT, C.ONNX_OUTPUT # "inputs", "fields"
C.ONNX_OPSET                # 17
C.data_path(...)            # path under DATA_DIR (/data in containers)
```

### Conventions

- Arrays are `[C, H, W] = [C, 64, 128]`; batches `[N, 3, 64, 128]`, float32.
- Row `j` is y with **row 0 = bottom wall**; column `i` is x with column 0 = inlet. Flow is left to right.
  `u > 0` is downstream, `v > 0` is upward. Plot with `origin="lower"`.
- Lattice units throughout. The inlet is `u = U0 = 0.05`, `v = 0`.
- `p` is a pressure coefficient: `(rho - rho_ref) / 3 / (0.5 * U0**2)`, where `rho_ref` is the mean density
  over the fluid cells of the outlet column.
- **Inside the obstacle, all three target channels are exactly 0.**
- `sdf`: distance in cells to the obstacle surface, **positive in the fluid, negative inside**.
- `mask`: 1.0 inside the obstacle, 0.0 in the fluid.
- `re_norm = (Re - 10) / 30`, a constant plane over the grid.

### Dataset (written by Aryan's generator)

`DATA_DIR/dataset.npz`:

| key | dtype / shape | meaning |
|---|---|---|
| `inputs` | float32 `[N,3,64,128]` | sdf, mask, re_norm (raw, unnormalised) |
| `targets` | float32 `[N,3,64,128]` | u, v, p (physical lattice units, 0 inside obstacle) |
| `cd_total`, `cl_total` | float32 `[N]` | solver's total force coefficients (not your concern) |
| `idx_train`, `idx_val`, `idx_test`, `idx_ood` | int64 | split indices into N |

`DATA_DIR/meta.json`: per sample `kind`, `params`, `re`, `d`, `blockage`, `iterations`, `converged`,
plus `n_attempted`, `n_dropped` and `split_counts`. The split is **stratified by (kind, Re bucket)**:
80/10/10 within each kind x 5-wide Re bucket, seeded. Triangles are only in `idx_ood`.

`DATA_DIR/norm.json` (computed on `idx_train` only):

```json
{"computed_on": "idx_train",
 "inputs":  {"channels": ["sdf","mask","re_norm"], "mean": [3 floats], "std": [3 floats]},
 "targets": {"channels": ["u","v","p"],            "mean": [3 floats], "std": [3 floats]}}
```

Use these stats for normalisation. Don't recompute your own. The fake dataset lives in `DATA_DIR/fake/`
with exactly the same files.

### Model handoff: `models/model.onnx` (the part that must be exact)

- **opset 17.**
- **One input named `"inputs"`**: float32 `[N, 3, 64, 128]`, **dynamic batch axis** (named `"batch"`),
  holding **RAW** inputs exactly as stored in `dataset.npz`, not normalised.
- **One output named `"fields"`**: float32 `[N, 3, 64, 128]` in **physical lattice units** (u, v, p),
  i.e. already denormalised, and **masked: all three channels multiplied by `(1 - mask)`**, where `mask`
  is input channel 1.
- So **normalisation, denormalisation and masking are inside the exported graph.** Wrap your network in an
  export module like this:

```python
class Exported(nn.Module):
    def __init__(self, net, in_mean, in_std, out_mean, out_std):
        super().__init__()
        self.net = net
        for k, v in dict(in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std).items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1))
    def forward(self, inputs):                       # raw [N,3,64,128]
        mask = inputs[:, 1:2]
        y = self.net((inputs - self.in_mean) / self.in_std)
        y = y * self.out_std + self.out_mean
        return y * (1.0 - mask)                      # all 3 channels zero inside the obstacle

torch.onnx.export(Exported(...).eval(), dummy, "models/model.onnx", opset_version=17,
                  input_names=["inputs"], output_names=["fields"],
                  dynamic_axes={"inputs": {0: "batch"}, "fields": {0: "batch"}})
```

- **Reference file:** `python -m windtunnel.dummy_model models/dummy.onnx` (in the train image, which has
  `onnx`) writes a trivial model with exactly these names, shapes and dynamic batch axis. Your export must
  load and run with the same onnxruntime call:
  `ort.InferenceSession(path).run(["fields"], {"inputs": x})[0]`.
- **torch-vs-ONNX check:** run the whole validation set through both the torch `Exported` module and
  onnxruntime. The **max absolute difference must be < 1e-4** (`C.TORCH_ONNX_MAX_ABS_DIFF`). Record it in
  the card. Also check it with batch size 1 and with a batch > 1 (to prove the dynamic axis works).
- **Commit `model.onnx` only if it is under 20 MB** (a 1-2M parameter float32 model is ~4-8 MB, so it should be).

### `models/model_card.json`

```json
{
  "param_count": 1234567,
  "config": {"base_channels": 32, "depth": 4, "...": "everything needed to rebuild the net"},
  "loss": {"type": "per-channel normalised MSE + w * mean(div^2)", "divergence_weight": 0.0},
  "ablation": {
    "no_divergence":   {"divergence_weight": 0.0,  "val": {"loss": 0.0, "rel_l2_u": 0.0, "rel_l2_v": 0.0, "rel_l2_p": 0.0}},
    "with_divergence": {"divergence_weight": 0.01, "val": {"...": "same keys"}},
    "chosen": "with_divergence",
    "reason": "lower val rel L2 on velocity"
  },
  "training": {"epochs_run": 0, "best_epoch": 0, "seconds": 0.0, "hardware": "e.g. RTX 3060 12GB / i7-12700",
               "seeds": {"torch": 0, "numpy": 0}, "dataset": "meta.json n, n_dropped, seed"},
  "validation": {"loss": 0.0, "rel_l2_u": 0.0, "rel_l2_v": 0.0, "rel_l2_p": 0.0},
  "onnx": {"opset": 17, "torch_vs_onnx_max_abs_diff_val": 0.0, "size_mb": 0.0},
  "reproduce": ["exact commands, in order"]
}
```

Validation relative L2 per channel: `||pred - true|| / ||true||` over fluid cells only (mask == 0),
computed on physical (denormalised) values. Aryan's `metrics.relative_l2` is the reference definition
once it lands. If yours differs, his wins.

## 5. Your tasks, in order

1. **Build the train image.** Copy `.env.example` to `.env` and set `DATA_DIR` to a folder on your machine.
   For CUDA, set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` and uncomment `gpus: all` in
   `compose.yaml` (or keep CPU). Then `docker compose --profile train build`. Check:
   `docker compose run --rm --entrypoint python train -c "import torch, onnx; print(torch.__version__, torch.cuda.is_available())"`.
2. **Write `windtunnel/model.py` and `scripts/train.py` against the fake dataset.**
   - Make the fake data: `docker compose run --rm --entrypoint python train scripts/make_fake_dataset.py`
     (writes `DATA_DIR/fake/`).
   - Small 2D U-Net, **~1-2M parameters**; print the count at startup. 64x128 divides cleanly by 2 four times.
   - `train.py` should take `--data` (default `C.data_path()`, or `C.data_path("fake")` for fakes),
     `--div-weight`, `--epochs`, `--seed`, `--out`.
3. **Run the full dataset generation** with Aryan's generator, unchanged, once he says it's verified (sync point 2):
   `docker compose run --rm gen --n <N> --workers <cores> --seed 0`. Target **~1500 converged samples**.
   Report how many were dropped (`n_dropped` in meta.json) and the wall time. Share `meta.json` and
   `norm.json` (and the npz, if Aryan asks) with Aryan.
4. **Train with and without the divergence term.** Run `w = 0` and a small sweep of `w > 0` (e.g. 1e-3, 1e-2, 1e-1),
   with the same seed and budget. Choose by **validation** metrics only. Put both arms in the card.
5. **Export to ONNX and write `model_card.json`** (section 4), including the torch-vs-ONNX check.
6. **Open a PR** from `abhay/<topic>` into `main`. Aryan merges.

## 6. Training guidance

- **Optimiser:** Adam (lr ~1e-3 to start), short cosine schedule to ~0.
- **Loss:** per-channel MSE on **normalised** targets (using `norm.json`), averaged over the 3 channels, plus
  `w * mean((du/dx + dv/dy)^2)`. Compute the divergence by central finite differences on **physical**
  (denormalised) u, v over interior fluid cells, and keep cells next to the obstacle and walls out of it.
  LBM is only weakly compressible, so this is a *soft* prior, not an exact constraint. Masking: compute the
  data loss either on all cells after multiplying the prediction by `(1 - mask)`, or on fluid cells only;
  say which in the card.
- **Early stopping** on validation loss (patience ~10-15 epochs). **Save the best checkpoint by validation
  loss** to `DATA_DIR/checkpoints/`.
- **Fixed seeds** (torch, numpy, python `random`; deterministic dataloader order). Report them.
- **Log** `epoch,train_loss,val_loss,lr,seconds` per epoch to `DATA_DIR/results/train_log.csv`. For the ablation,
  use one log per arm, e.g. `train_log_w0.csv`, plus the chosen one as `train_log.csv`.
- **Budget:** the whole thing should train in **under an hour** on your hardware. Report the actual time and hardware.
- Augmentation: vertical flip (y -> -y) is physically valid **only** if you also negate `v` and the
  geometry is mirrored. It's optional; if you use it, say so.

## 7. Working before real data exists

- `scripts/make_fake_dataset.py` writes a small dataset (~100 samples) with **the exact real schema**.
  Targets are analytical potential flow around circles; OOD holds triangles with a placeholder target.
  The physics is fake (no Re dependence, no walls). It exists to prove the pipeline.
- `windtunnel/dummy_model.py` has a numpy `DummySurrogate` and `export_dummy_onnx()`, a reference ONNX
  file with the exact I/O contract.
- **Do the whole loop on the fake data first:** train a few epochs, export, pass the torch-vs-ONNX check,
  and confirm the file loads with `onnxruntime` using the names above. Then rerun on the real data.

## 8. Branching and commits

- Branches named `abhay/<topic>` (e.g. `abhay/unet`, `abhay/export`). PRs into `main`. **Never push to `main`.**
- Plain, one-line, imperative commit messages ("Add U-Net model", "Log val loss per epoch").
- **No AI co-author lines** (`Co-authored-by: ...`) and no "Generated with ..." lines in commits or PRs.
- **No new dependencies without asking Aryan.** Dataset, checkpoints and logs go to `DATA_DIR`, never the repo.

## 9. Sync points

1. Contract, fakes and Docker committed (**done**).
2. Aryan's solver and generator verified on a small set; he tells you to run the full generation.
3. Your full dataset + stats (`meta.json`, `norm.json`, drop count, wall time) shared.
4. First `model.onnx` merged.
5. Final eval run (`scripts/evaluate.py`) and app demo.

## 10. Definition of done (your side)

- [ ] `models/model.onnx` and `models/model_card.json` merged into `main`.
- [ ] Ablation (with/without divergence) numbers in the card, choice made on validation only.
- [ ] torch-vs-ONNX max abs diff on the validation set < 1e-4, recorded in the card.
- [ ] The exact commands to reproduce (dataset seed, training, export) in the card and the PR description.
