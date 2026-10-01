# interactive-wind-tunnel (local notes, never committed)

Neural surrogate for steady 2D flow past a shape in a **simulated** channel (not a physical tunnel).
Input: geometry + Re. Output: u, v, p on a 128x64 grid. Shown in a Streamlit "virtual wind tunnel".
Ground truth is our own numpy LBM solver. 2D only, Re 10-40 only (steady regime).

## Split
- Aryan (me, weak laptop, never installs torch): contract, geometry, lbm, generate, metrics, baselines,
  surrogate (onnxruntime), evaluate, viz, app, tests, fake data, dummy model.
- Abhay: windtunnel/model.py, scripts/train.py, scripts/export_onnx.py, full dataset run, models/model.onnx + model_card.json.
- **In any conflict between my code and Abhay's, mine is the reference.** Details for him: ABHAY.md.

## Fixed decisions (ask Aryan before changing)
1. D2Q9 LBM, BGK, bounce-back obstacle + top/bottom walls, fixed-velocity inlet, zero-gradient outlet.
2. Grid NX=128, NY=64, arrays [C, 64, 128], row 0 = bottom, flow left to right. Obstacle at (32, 31.5).
3. U0=0.05, D = height across flow, 10 <= D <= 16 (blockage <= ~26%, stored per sample). nu=U0*D/Re, tau=3nu+0.5, reject tau<0.51.
4. Re 10-40. Steady when rel. change of u < 1e-5 over 100 steps; cap MAX_ITERS (provisional 40k). Non-converged dropped and counted.
5. Train kinds: circle, ellipse, rotated rectangle. NACA deferred. Triangles OOD only.
6. Inputs: sdf (+ fluid, - inside), mask, re_norm. Outputs: u, v, p; p = (rho - rho_ref)/3 / (0.5 U0^2), rho_ref = mean outlet density; all 3 channels 0 inside obstacle.
7. U-Net 1-2M params; loss = per-channel normalised MSE + w * mean(div^2), w chosen on val, ablation with/without.
8. Pressure drag/lift by surface integration, same estimator on GT and prediction. cd_total (momentum exchange) reported separately, never compared to it.
9. Baselines: mean field, nearest neighbour, potential flow (circles; unbounded-flow assumption noted).
10. Targets (hypotheses): velocity rel. L2 < 10%, pressure-drag error < 10%, < 50 ms/shape CPU onnxruntime, >= 100x faster than solver. OOD: no target.
11. Split stratified by (kind, Re bucket of width 5), 80/10/10, seeded. Test and OOD never used for tuning.

## Contract
All constants in windtunnel/contract.py. Dataset: DATA_DIR/dataset.npz (inputs, targets [N,3,64,128] f32; cd_total, cl_total [N];
idx_train/val/test/ood), meta.json, norm.json (train split only). ONNX: models/model.onnx, opset 17, input "inputs" raw [N,3,64,128],
output "fields" physical + masked, dynamic batch; torch-vs-ONNX max abs diff < 1e-4 on val. Loader: surrogate.load_surrogate(path).

## Open tasks for the solver/eval sessions
- Measure the surface-integration noise floor (Aryan asked: circles at a few Re, perturb, recompute drag). Note: the
  outlet-pressure offset cancels in a closed-surface integral, so also perturb surface pressure / sub-cell shift the circle.
- Confirm MAX_ITERS against real convergence times.
- Stratified split balances coverage but does not stop near-duplicate leakage; report NN-baseline distance to train.

## Environment
Windows + PowerShell, everything in Docker. Base image ~900 MB unpacked (accepted). DATA_DIR in .env = D:\interactive-wind-tunnel-data -> /data.
```
docker compose build                     # base only; never build the train profile
docker compose run --rm test
docker compose run --rm gen --n 50 --workers 2 --seed 0
docker compose run --rm eval
docker compose up app                    # localhost:8501
docker compose run --rm shell
docker compose run --rm --entrypoint python shell scripts/make_fake_dataset.py
```
In Git Bash, prefix docker commands with MSYS_NO_PATHCONV=1 when passing /data paths.

## Git
- Author: repo-local 145683044+Aryan-Pillai7@users.noreply.github.com. Never global.
- No Claude co-author / "Generated with" lines anywhere. Plain one-line imperative messages, small commits.
- `git ls-files` before every commit. Push to main allowed; never force-push. Abhay works on abhay/<topic> branches via PR.
