# Session context: stair-QAT experiments (as of 2026-10-29)

Handover for a new Claude Code session. Read this, then `QAT_UPDATE_RULE.md` (update rule, gradient modes, saturation mask).
Nothing from the PyTorch QAT work has been committed. Branch: `feature/validating_the_alpha_scheduling`. No jobs are running.

## 1. What the project is

Quantization-aware training (QAT) of small MLPs with fixed-point quantization: 8 bits total, 4 fractional (lsb = 0.0625, range [-8, 7.9375]).
The user compares a tunable, differentiable "stair" non-linearity (parameters alpha, beta) against hard STE and PTQ.
They are preparing findings for professors (plots, short reports, experiment descriptions in a fixed format). Python: `venv/bin/python`. Host: 12 cores.

**Report format the user asks for:** `Experiment Name/Code / Objective / Methods Used (Briefly) / Status (Completed or Ongoing)`. For short write-ups they want plain text in chat, not a `.md` file, unless they say otherwise.

**Pronouns:** use they/them for everyone (the user has not stated any).

## 2. Core concepts (what the user has been learning; explain consistently)

- **Two copies of each weight:** float master `w` (the only thing updated) and `wq = stair(w)` (what the network and loss see).
- **Stair geometry** (`src/quantization/torch_stair.py`, `stair_geometry`): around each grid point `g`, widths a = alpha*lsb/(alpha+1) (flat), b = lsb/(alpha+1) (steep), a+b = lsb; heights c = lsb/(1+beta*alpha) (flat), d = lsb-c, c+d = lsb. Slopes: flat c/a, steep d/b; for beta = 1 these are **1/alpha** and **alpha**. alpha = beta = 1 is the identity; large alpha approaches rounding. Zone test: `|w - round(w/lsb)*lsb| <= a/2` means flat ("trap"); the grid point is the centre of the flat zone. Forward: flat `g + diff*slope_flat`; steep `g + sign(diff)*[(a/2)*slope_flat + (|diff|-a/2)*slope_steep]`; then clamp to the range (`x_soft` is the value before the clamp, `wq` after it). Verified numerically this session: continuous, monotone, odd-symmetric, rises exactly one lsb per step, identity at alpha=beta=1, approaches rounding as alpha grows.
- **Update:** `w <- w - lr * dL/dwq * dwq/dw`. `dL/dwq` comes from backprop; `dwq/dw` is the stair slope (c/a or d/b) in the zone `w` sits in. The slope is the slope of the staircase, not of the loss curve. FP has slope 1.
- **Gradient modes** (`grad_mode`): `exact` = true slopes (1/alpha, alpha); `clamped` = **slope** (not gradient) clipped: flat to [0.1, 1], steep to [0, 1] (copied from the original NumPy `mlp.py`, hand-chosen, no documented derivation; the flat cap of 1 only bites for alpha<1, the steep cap bites for alpha>1, the flat floor 0.1 bites for alpha>10, all for beta=1); `ste` = slope 1 through the stair ("soft STE" / "slope 1" in the experiments). The user wrote to the professor that they clip the *slope*, not the gradient.
- **Hard STE:** real rounding in the forward pass from epoch 0, slope 1 backward, no alpha/beta; master weights updated with the gradient w.r.t. the rounded weights. The standard QAT baseline.
- **Saturation mask** (`mask_saturation`, default True): zeroes the slope where the forward output was clipped at +-range. A design choice (the NumPy code did not mask). How often anything saturates in trained ReLU nets has NOT been measured.
- **"Everything quantized" training** (`quantize_activations=True`): input, weights, biases, pre-activations, activations, output. Evaluation (`forward_exact`) always quantizes all of these with real rounding. The tanh experiments trained with weights/biases only.
- **Linear-regression code** (`src/models/linear_regression.py`): `fit_qat_non_linear` uses the clamped slope and updates float weights (alpha 0.1 to 100, beta also varies, cosine lr); the other QAT variants there overwrite the weights with the rounded values (no master copy).

## 3. Setup shared by almost every experiment

- Data (`src/data/dataset.py`, `generate_complex_dataset`): 4 features uniform [-1,1], 2000 samples, y = sin(3*pi*x0)+0.5*sin(6*pi*x1)+0.3*x2*x3+N(0,0.01^2); 60/20/20 split. Each seed redraws data, split and init.
- Model: [4,32,32,16,1], float64, Xavier N(0,1/fan_in) init, zero biases. Full-batch updates (one step per epoch, no momentum for SGD).
- Float training: tanh (50k epochs, lr 0.01) and ReLU (100k epochs, lr 0.01).
- **Start B** = best-validation checkpoint of the float run (used for all QAT/STE runs; one per seed, shared by every method). ReLU starts: `results/torch_qat/relu_lr0.01_100k/checkpoints/seed_XX.pt` (best epochs 58740, 47089, 28847, 48203, 64967 for seeds 0-4). Tanh starts: `results/torch_qat/e2/checkpoints/`. The ReLU lr 0.003 float run was never used as a start. (Start A, post-plateau-escape, was only used in E2.)
- alpha ramps linearly 1 -> alpha_end over 10,000 epochs ("transition").
- **Convergence rule:** nothing is judged during the ramp; afterwards the exact fixed-point validation MSE is tracked, improvement counts only if better by 1e-6, stop after 5000 epochs without improvement (patience), restore the best checkpoint. Cap 50k (tanh) / 100k (ReLU). Hard STE has no ramp, so the rule is active from epoch 0. Caveat found this session: with noisy validation the rule often stops right after the ramp and the best checkpoint sits just after epoch 10,000; "converged" means the rule fired, not that the weights stopped moving.
- **Diverged** = final exact validation MSE > 2.0 (`DIVERGED_MSE`). Constant-zero predictor is ~0.6; dead runs ended at 8-65. Runs still above 2.0 at epoch 60 can be aborted (`abort_val_above`, `abort_epoch=60`).
- Metrics: always the **exact fixed-point** validation/test MSE. Selection uses validation only; test is only reported.
- Storage: `experiments/torch_qat/storage.py` (`save_run`, `load_all`, `load_checkpoint`); each run = `.npz` (curves, states) + `.json` (`meta`, `scalars`). Plot functions are in `src/visualization/qat_plots.py`, style in `src/visualization/figure_style.py`.

## 4. Code changes this session (all uncommitted)

- `src/models/torch_qat.py`: `learning_rate_at` gained schedules `alpha_rising`, `warmup` (existing: `alpha_dependent` = lr0*(alpha_start/alpha)^power, so power 1 gives lr0/alpha); `train_qat` gained `abort_val_above`, `abort_epoch`, and **`optimizer_name="sgd"|"adam"`** (default sgd, so older runs are unchanged; Adam uses default betas). Tests: `venv/bin/python -m experiments.test_torch_qat` (17/17 pass; pytest is NOT installed).
- `requirements.txt`: added `optuna` (installed with the user's approval).
- `.gitignore`: raw/checkpoint dirs for e3c, relu*, relu_2x2, relu_grid, relu_optuna, relu_adam/raw.
- New runners/reporters in `experiments/torch_qat/`: `run_relu_fp.py`, `run_relu_2x2.py` + `report_relu_2x2.py`, `run_relu_grid.py` + `report_relu_grid.py`, `run_relu_optuna.py` + `report_relu_optuna.py`, `run_relu_adam.py` + `report_relu_adam.py`.
- `run_relu_adam.py`: methods `exact`, `soft_ste`, `hard_ste`; env `TASKS="method:lr:alpha[:inverse],..."` (explicit cells; `:inverse` means lr = lr0/alpha), `SEEDS`, `LRS`, `QAT_EPOCHS`, `PATIENCE`, `WORKERS`, `ADAM_DIR`. Run names: `seedXX_end_lr<lr>_<method>[_a<alpha>][_inv]_adam`. Skips finished runs. The loader in the report maps `lr_schedule == "lr0/alpha"` to schedule "inverse". Rebuild figures: `venv/bin/python -m experiments.torch_qat.report_relu_adam`.
- Docs written: `results/torch_qat/REPORT_RELU_FP.md` (ReLU float lr 0.01 vs 0.003), `results/torch_qat/e3c/REPORT_E3C.md`. The Adam report text was given in chat only (not saved); it says Adam did not rescue exact, which is no longer true for low-lr cells (see 6).

## 5. Experiments done (tanh phase, finished)

E1 (first QAT baselines; stopped on plateaus, so unreliable), E2 (warm-start from float, Starts A and B; hard STE vs clamped vs PTQ), E3a (lr sweep, Start B, alpha 1->10: hard STE best lr 0.1 test 0.1692; clamped best usable lr 0.03 by validation test 0.1940, lr 0.1 gives 0.1767 with 1/5 diverged; lr 0.3 diverged 5/5), E3b (final alpha sweep at lr 0.03: test 0.2006/0.2000/0.1940/0.1913/0.1902 for alpha 3/5/10/30/100), E3c (lr schedules, alpha 1->100: const 0.1 test 0.1720, rising 0.1833, falling 0.1861, warm-up 0.2080; hard STE 0.1692). Conclusion: hard STE strongest; clamped approaches it at lr 0.1; schedules do not beat a constant lr.

## 6. ReLU phase (everything quantized, Start B from the lr 0.01 / 100k ReLU float run)

ReLU float start (5 seeds): FP test 0.2078 +- 0.0257, PTQ of the start test 0.3627 +- 0.0571 (seeds 0-2 only: FP val/test 0.2014/0.2006, PTQ 0.3272/0.3577). ReLU float is noisy (validation +-0.03) and overfits late; lr 0.003 was worse (mean best val 0.257 vs 0.206). The FP "start" line is the best-validation checkpoint, not the end of training; it is a reference level, not a floor.

**2x2 (lr 0.01, alpha 1->10, SGD, 5 seeds), test MSE:** hard STE weights-only 0.1938, everything 0.1992; clamped weights-only 0.2715, everything 0.2600. Hard STE wins 5/5 in both scopes; the effect of quantizing activations is within noise (no clear improvement or harm).

**Grid (SGD, everything quantized, 5 seeds):** hard STE lr 0.01 / 0.03 / 0.1 / 0.2 test 0.199 / 0.185 / 0.163 (only 2 surviving seeds, 3/5 diverged) / all diverged. Clamped lr 0.01: alpha 3/10/30/100 test 0.305/0.260/0.263/0.268; lr 0.03: 0.251/0.308/0.416/0.499; lr >= 0.1 all diverged. Best clamped by validation: lr 0.01/alpha 100 (0.2611) tied with lr 0.03/alpha 3 (0.2634). Files: `results/torch_qat/relu_grid/`.

**Optuna (3 seeds 0-2, TPE, objective = mean validation over seeds, diverged = 10, finished):** clamped (20 trials; lr 0.005-0.06, alpha 1.5-100, log scale) best lr 0.0071, alpha 42, val 0.2432; top 6 trials within 0.004 (noise); good region lr 0.005-0.016, alpha 15-83; poor at lr >= 0.03 with high alpha. Hard STE (10 random trials; lr 0.01-0.08) best lr 0.0367, val 0.1663. Hard STE clearly beats the tuned clamped (gap ~0.077). Winners are optimistic (selected on the same 3 seeds); not yet evaluated on test over 5 seeds or on fresh seeds. Files: `results/torch_qat/relu_optuna/` (figs `fig_optuna_*.png`, `summary_relu_optuna.md`, `study.db`).

**Adam sweep (the professor's suggestion: exact slopes + Adam; STE as benchmark). 3 seeds, alpha 1->10 unless stated, 63 runs, files in `results/torch_qat/relu_adam/`:**
Professor's message (paraphrased): clipping the slope removes the point of the step function (large slope in the transition zone, ~zero in the flat zone lets the network reduce quantization error itself); optimisation difficulty can be handled by Adam; STE is fine as a benchmark; he hopes the proposed non-linearity outperforms it.

| method | lr -> val / test (mean of 3 seeds) |
|---|---|
| hard STE, Adam | 0.001: 0.200/0.217; 0.003: 0.195/0.208; 0.01: 0.201/0.207 |
| slope 1, Adam, alpha 10 | 0.001: 0.322/0.332; 0.003: 0.300/0.332; 0.01: 0.279/0.299; **0.03: 0.223/0.241**; 0.1: diverged 3/3 |
| exact, Adam, alpha 10 | 0.01: 0.556/0.553; 0.003: 0.441/0.431; 0.001: 0.325/0.340; 0.0003: 0.257/0.269; 0.0001: 0.246/0.260; **0.00003: 0.239/0.237** |

Final-alpha sweep: slope 1 at lr 0.01: alpha 3/10/30/100 val 0.358/0.279/0.256/0.249 (improves with alpha). Exact at lr 0.001: val 0.276/0.325/0.403/0.376 (worse with alpha; the dip at 100 is within seed spread). The alpha sweeps used one lr per method (chosen at alpha 10), so they are confounded by lr.
Inverse-schedule check (exact, Adam, lr = 0.003/alpha, alpha 30, ends at lr 0.0001): val 0.282, test 0.279 (per-seed val 0.323/0.242/0.280). **No constant-lr twin at alpha 30 exists** (lr 0.0001, alpha 30), so the schedule's effect is not isolated.
Reference on the same seeds: hard STE SGD lr 0.01 val/test 0.1785/0.1927; clamped SGD lr 0.01 alpha 100 0.2444/0.2539.

**Reading so far:** hard STE (SGD or Adam) is still best (val ~0.18-0.20). With learning rates in the right range both stair variants approach clamped-SGD level (best slope 1 cell 0.223 val / 0.241 test; best exact 0.239 / 0.237, flattening with lr; seed 1 of the 3e-5 cell drives most of that gain). Exact is worse than slope 1 at equal lr and gets worse with alpha. In many stair runs validation is lowest early (alpha small, epochs ~500-5000) and degrades as alpha grows; training loss rises with alpha (0.04 -> 0.10-0.25) and does not recover; this is untested as a cause. The inverse-schedule curves show flat step-like stretches in both validation and training loss (reason unknown; maybe weights stop moving, not checked). All new runs converged by the rule (15k-29k epochs), none hit the 150k cap.

## 7. Open questions and suggested next steps

1. Run the missing twin: exact, Adam, constant lr 0.0001, alpha 30 (3 runs, ~10 min), so the lr0/alpha result can be interpreted. Command pattern: `TASKS="exact:0.0001:30" QAT_EPOCHS=150000 venv/bin/python -m experiments.torch_qat.run_relu_adam` (nohup, log to `results/torch_qat/relu_adam/logs/`).
2. Float-continuation control: continue float training from Start B with the same stopping rule/checkpoint selection (no quantization). Hard STE's validation drops 0.206 -> 0.179 versus the float start, but test only 0.208 -> 0.199 (better on seeds 2 and 4 only); this is consistent with selection on a noisy curve and extra training. Not tested whether quantization acts as a regulariser. (~5 short runs.)
3. Evaluate the Optuna winners (clamped lr 0.0071/alpha 42; hard STE lr 0.0367) on test over all 5 seeds; fresh seeds would need a new ReLU float run (~11 min).
4. Hard-to-resolve confounds worth fixing: slope-1 never tried at lr 0.03 for alpha 30/100; exact at lr 0.0001 for alpha != 10; best-cell comparison gives exact more cells than hard STE.
5. Count saturation (weights, pre-activations, activations beyond +-8) and optionally rerun one cell with `mask_saturation=False`.
6. Ideas not run: stage-3 falling `1/alpha` schedule with SGD on ReLU, normalization (professor's idea; user deferred it), beta sweep (beta has always been 1 in PyTorch runs), slower alpha ramp (e.g. 30k epochs; user's "Part C"), Adam betas/constant-lr variants, raw (not rolling-median) validation plots for the flat-stretch behaviour.
7. Text to refresh: the Adam report written in chat predates the low-lr and schedule results.

## 8. Practical notes

- Long runs: `nohup env ... venv/bin/python -m experiments.torch_qat.<script> > results/torch_qat/<dir>/logs/<name>.log 2>&1 &`; 9-10 parallel workers, 10-25 min per batch for 100k-epoch ReLU runs. Smoke-test with `QAT_EPOCHS=300 PATIENCE=50 ADAM_DIR=/tmp/...`.
- Figures follow one visual language: fixed colour/marker per method, references as grey dotted/dashed lines, legend text outside the plot area; always look at rendered PNGs (label collisions happened several times).
- The user prefers to be asked before launches that change scope (several tool calls were interrupted for this), wants run counts and time estimates before long jobs, and wants plain answers about what is and is not tested. Do not claim results from E1/E2 numbers that are not in the notes without reading the summaries first (`results/torch_qat/e2/summary*.md`).
- Several claims above are hypotheses, not findings: ReLU noise from unit flips, compounding clamped-slope shrinkage across layers, "quantization as regulariser", why exact degrades with alpha.
