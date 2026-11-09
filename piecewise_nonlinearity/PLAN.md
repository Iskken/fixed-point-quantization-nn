# Plan review and revised plan (2026-11-09)

**Status: proposed.** The [README](README.md) plan will be updated once we agree on the changes below. None of them affect the code of steps 4–6 (shared quantizer, STE, staircase, tests).

## Summary

The core plan holds. ResNet-20 on CIFAR-10 is the anchor; we compare our pure staircase with the STE, EWGS and DSQ under one protocol, and reproduce published EWGS and DSQ numbers to show our implementations are right.

Measured timings show that the plan as written needs about 117 GPU hours, against about 110 available before Nov 20. The changes below bring it to about 63 hours. They also fix three methodological gaps:
- the same seeds are used for tuning and for results;
- EWGS and DSQ get no hyperparameter search;
- the win criterion is not fixed in advance.

## Where we are

- **Steps 1–3 done:** folder structure, CIFAR-10/100 download, and three float ResNet-20 models (seeds 0–2). Their test accuracy is 91.17 / 91.16 / 91.23 (see [pretrained/](pretrained/)), which matches the papers' float baselines: EWGS 91.4, DSQ 90.78.
- **Steps 4–6 done:** the shared quantizer, the STE, the pure staircase (`exact` or slope-1 backward, α schedule) and 23 passing sanity tests.

## What holds

- **The step 4–6 code is complete for step 7.** BatchNorm re-estimation (`set_force_hard`), the weights-only and activations-only switches (`w_bits` / `a_bits = None`) and the backward switch are already in place.
- **Our evaluation matches EWGS.** We score the model after the last epoch, with no selection. The official EWGS code reports the last checkpoint too (`CIFAR10/train_quant.py`, "Test accuracy @ last checkpoint").
- **The QAT pipeline works on real data.** Starting from the seed-0 float model, validation accuracy at 4/4 is 86–88% after only 50 training steps.
- **The training settings follow EWGS:** Adam, cosine decay, weight decay 1e-4, quantizer parameters at learning rate 1e-5, batch size 256.

## Measured cost

Measured on 2026-11-09 on our RTX 3050 Laptop GPU: ResNet-20, 4/4, fp32, batch size 256, starting from the seed-0 float model. Times are per epoch of 176 batches, plus one validation pass of about 1 s.

| Method | Time per batch | Time per epoch | Peak memory |
|---|---|---|---|
| STE | 102 ms | ~19 s | 1.3 GB |
| Staircase, exact backward | 142 ms | ~26 s | 1.3 GB |
| Staircase, slope-1 backward | 133 ms | ~24 s | 1.3 GB |

Two training jobs on the GPU at the same time run at 308 ms per batch each, against 142 ms alone. That is slower in total, so runs must go one at a time.

## Necessary changes

### 1. Fix the QAT length now, and tune at that same length

**Problem.** If we tune on short runs and run step 10 at a different length, the tuned settings may not carry over: the α ramp and the learning-rate decay both stretch with the run length.

**Change.** All runs in steps 7 and 10 use **60 epochs** (cosine decay to 0, α ramped linearly over the first half). Only the step 9 reproductions use the papers' 400 epochs.

**Cost.** About 19 min per STE run and 26 min per staircase run.

### 2. Tune one thing at a time, so tuning fits one night

**Problem.** The full grid (3 learning rates × 3 α × 3 β for the staircase, 3 learning rates for the STE, at 2/2 and 4/4) takes about 25 hours.

**Change.** All tuning runs use the development seed (seed 0).
- **At 2/2:**
  1. Tune the learning rate ∈ {3e-4, 1e-3, 3e-3} for the STE, and for the staircase at α_end = 2, β = 2.
  2. At the staircase's best learning rate, run the other 8 combinations of α_end ∈ {1.5, 2, 3} × β ∈ {1, 2, 3}.
- **At 4/4:** tune only the learning rate for both methods, reusing the α and β chosen at 2/2.

The 4/4 rule is fixed now, before any results.

**Selection metric.** Validation accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation.

**Also logged per run:**
- the gap between staircase-forward and exact-rounding accuracy;
- the share of weights that change level per epoch;
- the run time.

**Cost.** About 8 hours: 5.7 h at 2/2 and 2.3 h at 4/4.

### 3. Use separate seeds for tuning and for results

**Problem.** Our rule is "tune on development seeds, claim on fresh seeds", but float models exist only for seeds 0–2. Tuning on seed 0 and reporting on seeds 0–2 breaks the rule. The staircase is also tuned over more settings than the STE, so selection noise on the tuning seed favors it. Fresh seeds remove that bias.

**Change.** Train float models for **seeds 3 and 4**. Tune on seed 0 and report all step 10 results on **seeds 1–4**. Each run uses the float model and QAT seed with the same number.

**Cost.** About 1 hour per float model.

### 4. Give EWGS and DSQ the same learning-rate search

**Problem.** Only the STE and the staircase are tuned. A reviewer would call the baselines under-tuned.

**Change.** On seed 0, tune the learning rate ∈ {3e-4, 1e-3, 3e-3} for EWGS and DSQ at 2/2 and 4/4. Their other settings follow the papers:
- EWGS: δ from the Hessian, updated every 10 epochs;
- DSQ: α starts at 0.2 and is learned, and l and u are learned.

**Cost.** 12 runs, about 5 hours (estimated).

### 5. Fix the metric and the win rule before step 10

**Primary metric.** Test accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation. Re-estimation recomputes the BatchNorm statistics with exact rounding on 50 training batches, done the same way for every method. We also report the accuracy without re-estimation, which is what the papers report.

**Why re-estimation.** The staircase trains with the soft curve S, so its BatchNorm statistics do not match exact rounding. The MLP study had no BatchNorm, so this problem is new on ResNet-20.

**Proposed win rule,** per bit width, over the 4 fresh seeds. The staircase beats a method if:
1. its mean test accuracy is higher;
2. a paired t-test gives p < 0.025 (0.05 with a Bonferroni correction for the 2 bit widths);
3. it wins on at least 3 of the 4 seeds.

Otherwise we report a tie or a loss as such. With 4 seeds a Wilcoxon test cannot reach p < 0.05, so the t-test is the formal test, and the per-seed results are reported alongside it.

### 6. Correct the step 9 reproduction targets

- **EWGS at 1/1:** STE 84.7 and EWGS 85.6 (EWGS Tables 4–5). Same network (option-A ResNet-20) and same 400-epoch protocol, so this is a direct check.
- **DSQ at 2 bits:** DSQ Table 4 is "2-bit uniform quantization of ResNet-20 on CIFAR-10":

  | Variant | Top-1 |
  |---|---|
  | Standard quantization (STE) | 86.63 |
  | Fixed α | 86.95 |
  | Learned α | 87.25 |
  | Learned α, l, u | 88.44 |

  This gives a same-paper STE baseline at our main bit width. We run both the STE and DSQ with learned α, l, u, using EWGS's 400-epoch protocol, and compare **DSQ's gain over the STE (+1.81)**, not its raw accuracy. Three reasons:
  - the paper does not say whether both weights and activations are 2-bit;
  - its non-binary ResNet-20 seems to use 1×1 downsampling convs, since it discusses quantizing "downsampling convolution layers";
  - it gives no training schedule.
- **DSQ at 1/1 (84.11) becomes optional.** It is the "Piecewise DSQ" row of DSQ Table 3, a binarized setup with a type-A shortcut and the BNN activation replacement. That differs from our {0, 1} activation quantizer.
- **Naming.** DSQ calls one of its own variants "piecewise DSQ", so our method is always called the "piecewise-linear staircase".

### 7. Set priorities and drop 3/3

**Problem.** Everything does not fit (see the budget below).

**Change.** Drop 3/3; neither paper reports it on CIFAR-10. Run step 10 in this order:
1. ResNet-20 at 2/2 and 4/4: STE, staircase, EWGS and DSQ, each on 4 fresh seeds.
2. Ablations at 2/2, on 4 fresh seeds: staircase with slope-1 backward; weights only and activations only for both the STE and the staircase.
3. MobileNetV2 at 4/4: STE vs staircase, 3 seeds, using ResNet-20's tuned settings.
4. CIFAR-100, only if time is left.

## Compute budget

Totals are in GPU hours, for one job at a time. "Measured" uses the per-epoch times above; "estimated" assumes EWGS and DSQ cost about as much as the staircase, and that MobileNetV2 QAT costs about 1.5 times our earlier placeholder measurement.

| Item | As written | Revised | Basis |
|---|---|---|---|
| Step 7 tuning | 25 h (full grid, 60 epochs) | 8.0 h | measured |
| EWGS / DSQ learning-rate tuning | — | 5.2 h | estimated |
| Step 9 reproductions (400 epochs) | 10 h | 10.0 h | partly estimated |
| Float models, seeds 3–4 | — | 1.8 h | measured |
| Step 10, ResNet-20 main comparison | 19 h (4 methods × 3 bit widths × 4 seeds) | 12.9 h (2 bit widths) | partly estimated |
| Step 10 ablations | 11 h | 7.6 h | measured |
| MobileNetV2 (float + QAT) | 33 h (4 methods) | 17.4 h (STE vs staircase) | estimated |
| CIFAR-100 | 12 h | optional | estimated |
| **Total** | **~117 h** | **~63 h** | |

**Available:** about 110 hours. That assumes the laptop trains about 10 hours on each of the 11 nights from Nov 9 to Nov 19. The revised plan leaves room for reruns.

## Revised schedule

| Date | Day (coding, CPU) | Night (GPU) |
|---|---|---|
| Nov 9 | Float seeds 3–4 on the GPU; write `train_qat.py` | Step 7 tuning (~8 h) |
| Nov 10 | Step 8: EWGS and DSQ, with CPU tests; the two can be split between the two of us | EWGS/DSQ learning-rate tuning (~5 h) |
| Nov 11 | Read the tuning results, decide the paper's framing | Step 9 reproductions (~10 h) |
| Nov 12–13 | — | Step 10 main comparison (~13 h) |
| Nov 14–16 | Write the abstract (due Nov 16) | Step 10 ablations (~8 h) |
| Nov 16–20 | — | MobileNetV2, then CIFAR-100 if time is left |

## Decisions to confirm

1. QAT length of 60 epochs for steps 7 and 10; 400 epochs for step 9 only.
2. One-at-a-time tuning, with 2/2's α and β reused at 4/4.
3. Seed 0 for tuning; seeds 1–4 for results.
4. The same learning-rate search for EWGS and DSQ.
5. The primary metric (with BatchNorm re-estimation) and the win rule.
6. The step 9 targets: EWGS 1/1 and DSQ Table 4; DSQ 1/1 optional.
7. 3/3 dropped, and the priority order of step 10.

Once these are agreed, the README is updated in its plan table, rules, numbers to reproduce, compute budget and status.

## Sources

- **Timings:** measured on 2026-11-09 with a throwaway script (not committed), using the code at commit `80158ed`.
- **Float results:** [results/float/](results/float/) and [pretrained/README.md](pretrained/README.md).
- **EWGS evaluation and training settings:** the official code, `CIFAR10/train_quant.py` and `train_fp.py` (see "Reference code" in the README for the pinned version).
- **DSQ settings:** the DSQ paper (Gong et al., ICCV 2019), Section 4.1, Section 4.3 and Tables 3–4.
- **MLP study:** [results/torch_qat/study_claude/STUDY_LOG.md](../results/torch_qat/study_claude/STUDY_LOG.md).
