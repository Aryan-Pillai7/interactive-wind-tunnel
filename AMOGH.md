# AMOGH.md: handoff of the model half (from Abhay)

You are picking up the model side described in [ABHAY.md](ABHAY.md). Read that first: the rules,
the contract and the ownership table there still apply to you. This file says **what is already done,
what we found, and exactly what is left**.

## 1. Status at a glance

| ABHAY.md task | Status |
|---|---|
| 1. Build the train image | Done once (CPU, torch 2.7.1, onnx 1.18). Image was deleted afterwards; rebuild it. |
| 2. `model.py` + `train.py` against the fake dataset | **Done**, tested. |
| 3. Full dataset generation | **Blocked**: Aryan's `lbm.py` / `generate.py` are still stubs (sync point 2 not reached). |
| 4. Divergence ablation | Code done. Full sweep on fake data **not finished** (stopped after the `w = 0` arm). |
| 5. ONNX export + `model_card.json` | Code done and verified on a 3-epoch smoke run. Real `models/model.onnx` **not produced yet**. |
| 6. PR into `main` | Open from branch `abhay/unet` (code only, no model file yet). |

Nothing trained has been committed. `models/` does not exist in the repo yet.

## 2. What is in the code

- **`windtunnel/model.py`**: `UNet` (default `base_channels=16, depth=4`, **1,942,611 params**).
  Each level is `(3x3 conv, GroupNorm(8), GELU) x 2`, maxpool down, 2x2 transposed conv up, concat skips,
  1x1 head. 64x128 halves four times to a 4x8 bottleneck. Also `Exported`, the contract wrapper from ABHAY.md
  (normalise, run the net, denormalise, multiply by `1 - mask`).
  `python -m windtunnel.model` prints the config and parameter count.
- **`scripts/train.py`**: trains one arm per `--div-weight` value (same seed and budget for each), then picks
  the arm with the lowest **validation** relative L2 on velocity (stacked u, v).
  - Reads only `idx_train` and `idx_val`. `idx_test` / `idx_ood` appear nowhere in the code (a test checks this).
  - Data loss: per-channel MSE on `norm.json`-normalised targets over **fluid cells only**, then the mean over u, v, p.
  - Divergence term: central differences on **physical** u, v; cells next to the obstacle (3x3 dilation),
    rows 0-1 and NY-2..NY-1, and the first and last columns are excluded.
  - Adam, cosine LR to 0 over `--epochs`, early stopping on val loss (`--patience 15`), fixed seeds,
    deterministic batch order. Vertical-flip augmentation exists behind `--flip-aug` (off by default).
  - Writes to `OUT` (default = `--data`): `checkpoints/best_w<w>.pt`, `checkpoints/best.pt` (chosen arm),
    `results/train_log_w<w>.csv`, `results/train_log.csv` (chosen arm), `results/ablation.json`.
- **`scripts/export_onnx.py`**: loads `best.pt`, exports opset 17 with dynamic `batch` axis, checks I/O names
  and shapes, checks that **masked cells are exactly 0**, runs **the whole val set** through torch and onnxruntime
  at batch 1 and batch 8 and **asserts max abs diff < 1e-4**, then writes `models/model.onnx` and
  `models/model_card.json` (all keys from ABHAY.md section 4, plus the full sweep and the reproduce commands).
- **`tests/test_model.py`**: 5 tests (param budget, shapes + masking, divergence on a linear field + the
  exclusion mask, no test/OOD access in `train.py`, ONNX round trip at batch 1 and 4). They skip in the base image
  (no torch). Run them in the train image (see section 5).

## 3. Results so far (fake data only, CPU)

Fake data = `scripts/make_fake_dataset.py --seed 0`: 104 samples, train 76 / val 10 / test 10 / OOD 8.
Remember the fake physics is potential flow (no Re dependence, no walls), so these numbers mean
"the pipeline works", not "the model is good".

| Run | Result |
|---|---|
| 3-epoch smoke run, export | torch vs ONNX max abs diff on val **1.19e-6** (batch 1 and 8); ONNX size **7.47 MB** |
| Full run, `w = 0`, 100 epochs max | best epoch 87; val rel L2 **u 0.0049, v 0.0436, p 0.0459**, vel 0.0062 |
| Full run, `w = 0.01` | stopped at epoch 15 by request; other arms not run |

Speed: about 1.3 s/epoch in the CPU Docker image (16 threads) on the fake set.

## 4. Things you must know (also raise them with Aryan)

1. **The divergence weights suggested in ABHAY.md (1e-3 to 1e-1) do nothing.** In physical lattice units,
   velocity gradients are about U0/D ~ 1e-3, so `mean(div^2)` is about **1e-6 to 1e-8**: about 1e-6 on the fake
   ground truth and 1.4e-8 on the trained `w = 0` model. The data loss is O(1e-3 to 1). At `w = 0.1` the penalty
   is about 1e-7 of the total. This is why the default sweep in `train.py` is **`0, 1e-2, 1, 1e2, 1e4`**.
   Do not change the definition (physical units) without Aryan agreeing; if you want a scale-free version
   (divide by U0), propose it in the PR. Re-check the scale on the real LBM data, which is not divergence-free.
2. **Real data does not exist yet.** Only Aryan can unblock this (he must say his solver is verified).
   Until then, everything runs on `DATA_DIR/fake`.
3. Validation relative L2 here is our own implementation (`relative_l2_batch` in `train.py`, per sample, then
   averaged). `metrics.relative_l2` is still a stub. When Aryan's lands, **his definition wins**: compare
   the two and switch if they differ.
4. **Windows / Git Bash gotchas:**
   - In `.env` use forward slashes: `DATA_DIR=C:/interactive-wind-tunnel-data`. A backslash path gets mangled.
   - Git Bash rewrites `/data/...` and `/tmp/...` arguments into Windows paths. Prefix docker commands with
     `MSYS_NO_PATHCONV=1`, or use PowerShell.
   - The first `docker compose --profile train build` died with `rpc error ... EOF` (the Docker daemon dropped
     mid-download). Re-running it worked.
   - Stopping the shell running `docker compose run` does **not** stop the container. Use `docker stop`.
5. **Rules from ABHAY.md that are easy to miss:** never load `idx_test` / `idx_ood` in training or tuning;
   no new dependencies without asking Aryan; datasets, checkpoints and logs go to `DATA_DIR`, never the repo;
   one-line imperative commit messages; **no AI co-author or "Generated with" lines**; never push to `main`.

## 5. What is left: do it in this order

### A. Set up (about 10 min plus the image download)

```bash
git fetch origin && git checkout abhay/unet     # or branch off it: amogh/<topic>
cp .env.example .env                            # set DATA_DIR=C:/your/data/folder (forward slashes)
docker compose --profile train build
docker compose run --rm --entrypoint python train -c "import torch, onnx; print(torch.__version__, torch.cuda.is_available())"
docker compose run --rm --entrypoint python train -m pytest -q -p no:cacheprovider tests/test_model.py   # expect 5 passed
```

For a GPU, set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` in `.env` and uncomment `gpus: all`
in `compose.yaml`. Say so in your PR, since it changes a file in the train section.

### B. Finish the loop on fake data (proves the pipeline end to end)

```bash
export MSYS_NO_PATHCONV=1   # Git Bash only
docker compose run --rm --entrypoint python train scripts/make_fake_dataset.py
docker compose run --rm train --data /data/fake --epochs 100 --seed 0
docker compose run --rm export --data /data/fake
```

Expect 5 arms. On CPU this takes about 10-20 minutes in total. Then check that:
- [ ] `export` printed a max abs diff < 1e-4 for batch 1 and batch 8 (it asserts this).
- [ ] `models/model.onnx` is under 20 MB (about 7.5 MB expected).
- [ ] `models/model_card.json` has `ablation.no_divergence`, `ablation.with_divergence`, `ablation.sweep`,
      `validation`, `onnx`, `reproduce`.
- [ ] Optionally, the same onnxruntime call as the app works:
      `ort.InferenceSession("models/model.onnx").run(["fields"], {"inputs": x})[0]`.

A fake-data `model.onnx` may be committed as a pipeline check, but say clearly in the PR that it was
**trained on fake data**. The real one replaces it later.

### C. When Aryan says the solver is verified (sync point 2)

```bash
docker compose run --rm gen --n <N> --workers <cores> --seed 0     # target ~1500 converged samples
```
- [ ] Record the wall time and `n_dropped` from `meta.json`; share `meta.json` + `norm.json` with Aryan (sync point 3).
- [ ] Re-check the divergence scale on real data (section 4.1) and adjust the sweep if needed (`--div-weight ...`).

### D. Real training, export, PR

```bash
docker compose run --rm train --seed 0                              # defaults: --data /data, 150 epochs, 5 arms
docker compose run --rm export --hardware "<GPU / CPU of your machine>"
```
- [ ] The whole sweep finishes in **under an hour** (ABHAY.md budget). If not, cut `--epochs` or the number of arms and report it.
- [ ] Choose on **validation only**. The script does this; don't override it by looking at test numbers.
- [ ] Commit `models/model.onnx` (if < 20 MB) and `models/model_card.json`.
- [ ] PR into `main`: paste the card's `reproduce` commands, the ablation table, hardware and time, and
      the issues from section 4.

### E. Definition of done (from ABHAY.md section 10)

- [ ] `models/model.onnx` and `models/model_card.json` merged into `main`.
- [ ] Ablation (with and without divergence) in the card, chosen on validation only.
- [ ] torch-vs-ONNX max abs diff on val < 1e-4 recorded in the card.
- [ ] Exact reproduce commands (dataset seed, training, export) in the card and the PR description.
