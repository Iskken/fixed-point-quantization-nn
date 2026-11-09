# Plan review and revised plan (2026-11-09)

**Status (2026-11-09):** decisions 1–4 agreed, decision 5 partly (see [Decisions to confirm](#decisions-to-confirm)). The [README](README.md) plan will be updated once all are agreed. Decision 5 added BatchNorm re-estimation to the step 4–6 code (`models/quant_layers.reestimate_bn`, `datasets.get_bn_loader`); nothing else in steps 4–6 changed.

**Deadline constraint.** The abstract is due on **Nov 16** and **cannot be edited** after submission, so the work is ordered by what the abstract needs (see [Abstract deadline](#abstract-deadline-nov-16-not-editable)).

## Summary

The core plan holds. ResNet-20 on CIFAR-10 is the anchor; we compare our pure staircase with the STE, EWGS and DSQ under one protocol, and reproduce published EWGS and DSQ numbers to show our implementations are right.

Measured timings show that the plan as written needs about 117 GPU hours, against about 110 available before Nov 20. The changes below bring it to about 75 GPU hours. That includes a 2.6-hour pilot that fixes the training length, and an equal tuning budget for all four methods. The work is shared between two laptops (RTX 3050 with 4 GB and with 6 GB), which roughly halves the wall-clock time. The changes also fix three methodological gaps:
- the same seeds are used for tuning and for results;
- EWGS and DSQ get no hyperparameter search;
- the win criterion is not fixed in advance.

Because the abstract cannot be edited after Nov 16, about 50 of those 75 GPU hours (about 25 on each laptop) are scheduled before it. The rest goes into the paper only.

## Where we are

- **Steps 1–3 done:** folder structure, CIFAR-10/100 download, and three float ResNet-20 models (seeds 0–2). Their test accuracy is 91.17 / 91.16 / 91.23 (see [pretrained/](pretrained/)), which matches the papers' float baselines: EWGS 91.4, DSQ 90.78.
- **Steps 4–6 done:** the shared quantizer, the STE, the pure staircase (`exact` or slope-1 backward, α schedule) and the sanity tests (29 passing, including BatchNorm re-estimation).

## What holds

- **The step 4–6 code is complete for step 7.** BatchNorm re-estimation (`reestimate_bn` with `get_bn_loader`, see change 5), the weights-only and activations-only switches (`w_bits` / `a_bits = None`) and the backward switch are already in place.
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

### 1. Fix the QAT length with a pilot, and tune at that same length

**Problem.** If we tune on short runs and run step 10 at a different length, the tuned settings may not carry over: the α ramp and the learning-rate decay both stretch with the run length. A budget that is too short would also measure how fast each method converges rather than how good it gets. That would penalize the staircase in particular, because it spends the first half of training ramping α.

**Change.** All runs in steps 7 and 10 use one fixed budget of N epochs, with cosine decay to 0 and α ramped linearly over the first half of training. Only the step 9 reproductions use the papers' 400 epochs. N is chosen by a pilot, with a rule fixed before the pilot runs:

- **Pilot.** STE and the staircase (α_end = 2, β = 2, learning rate 1e-3) at 2/2, the hardest setting, on the development seed (seed 0). Each is trained for 30, 60 and 120 epochs, each run with its own cosine schedule.
- **Rule.** N is the shortest budget in {30, 60, 120} such that going from N to 2N changes
  - (a) each method's validation accuracy by less than 0.3 points, and
  - (b) the gap between the methods (staircase − STE) by less than 0.3 points.

  Condition (b) is the important one: our conclusion must not depend on the budget. Condition (a) makes sure the absolute numbers are close to convergence, so that they are meaningful next to the papers'. A change in which method is ahead counts only through (b): a flip from +0.1 to −0.1 is noise, not a change of ranking.
- **Why 0.3 points.** The threshold must not be smaller than run-to-run noise, or the rule could not be tested with one run per budget. It must also be clearly smaller than the effects we want to measure, or the budget could matter as much as the method. 0.3 satisfies both:
  - **Noise:** our three float models differ on the validation set with a standard deviation of 0.12 points (91.66, 91.64, 91.86), so 0.3 is about 2.5 standard deviations.
  - **Effect size:** 0.3 is below half of the method improvements reported on this benchmark: +0.9 for EWGS over STE at 1/1 (EWGS Table 5) and +0.62 for DSQ's α mechanism at 2 bits (DSQ Table 4).
- **Fallback, fixed now.** If even 60 → 120 fails the rule, N = 120. Tuning then takes two nights, and for the abstract EWGS and DSQ move from "should have" to "if time allows". If 30 → 60 passes, N = 30 and every later step costs half as much.
- **Limitation.** There is one run per budget, so the pilot detects large budget effects, not small ones. The paper says so.

**Cost.** The pilot takes about 2.6 GPU hours: 210 epochs per method, at about 19 s (STE) and 26 s (staircase) per epoch. At N = 60, a run takes about 19 min (STE) or 26 min (staircase).

**In the paper** (N and the largest observed change are filled in from the pilot):

> We fixed the fine-tuning budget in advance with a rule applied on the development seed: the shortest budget in {30, 60, 120} epochs for which doubling it changed each method's validation accuracy, and the accuracy gap between methods, by less than 0.3 points. This threshold is about 2.5 standard deviations of the seed-to-seed variation of our full-precision models on the validation set (0.12 points), and below half of the method improvements reported on this benchmark (+0.9 for EWGS over STE at 1/1 [EWGS, Table 5]; +0.62 for DSQ's α mechanism at 2 bits [DSQ, Table 4]). To validate our EWGS and DSQ implementations, we additionally reproduce published results under the papers' original 400-epoch protocol.

### 2. Tune one thing at a time, so tuning fits one night

**Problem.** The full grid (3 learning rates × 3 α × 3 β for the staircase, 3 learning rates for the STE, at 2/2 and 4/4) takes about 25 hours.

**Change.** All tuning runs use the development seed (seed 0).
- **At 2/2:**
  1. Tune the learning rate ∈ {3e-4, 1e-3, 3e-3} for the STE, and for the staircase at α_end = 2, β = 2.
  2. At the staircase's best learning rate, run the other 8 combinations of α_end ∈ {1.5, 2, 3} × β ∈ {1, 2, 3}. At the STE's best learning rate, run its 8 settings from change 4.
- **At 4/4:** tune only the learning rate for both methods, reusing the settings chosen at 2/2.

The 4/4 rule is fixed now, before any results.

**Selection metric.** Validation accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation.

**Also logged per run:**
- the gap between staircase-forward and exact-rounding accuracy;
- the share of weights that change level per epoch;
- the run time.

**Cost.** About 10.5 hours at 60 epochs for the STE and the staircase together, 14 runs each: 4.4 h for the STE and 6.1 h for the staircase, on one laptop. EWGS and DSQ follow the same procedure on the other laptop (change 4). If the pilot picks a different budget, this scales in proportion.

### 3. Use separate seeds for tuning and for results

**Problem.** Our rule is "tune on development seeds, claim on fresh seeds", but float models exist only for seeds 0–2. Tuning on seed 0 and reporting on seeds 0–2 breaks the rule. The staircase is also tuned over more settings than the STE, so selection noise on the tuning seed favors it. Fresh seeds remove that bias.

**Change.** Train float models for **seeds 3 and 4**. Tune on seed 0 and report all step 10 results on **seeds 1–4**. Each run uses the float model and QAT seed with the same number.

**Cost.** About 1 hour per float model.

### 4. Give every method the same tuning budget (option C)

**Problem.** Only the STE and the staircase are tuned, so a reviewer would call the baselines under-tuned. Tuning only the learning rate of EWGS and DSQ would still leave the staircase with more tuning than the others.

**Change.** Every method gets the same budget of **14 tuning runs** on the development seed (seed 0), with the procedure of change 2:
- **At 2/2:** 3 runs over the learning rate {3e-4, 1e-3, 3e-3} with the method's default settings, then 8 runs over its own settings at the best learning rate.
- **At 4/4:** 3 runs over the learning rate, keeping the settings chosen at 2/2.
- **Grid edge:** if the best learning rate is at the edge of the grid, the grid is extended by one step in that direction, for every method alike.

The 8 settings come from each paper's own ablations, so no ranges are invented:

| Method | Default (run first) | Its 8 other settings at 2/2 | Source of the range |
|---|---|---|---|
| Staircase | α_end = 2, β = 2 | the other 8 of α_end ∈ {1.5, 2, 3} × β ∈ {1, 2, 3} | our MLP study |
| STE | quantizer learning rate 1e-5, weight decay 1e-4 | the other 8 of quantizer learning rate {1e-6, 1e-5, 1e-4} × weight decay {0, 1e-4, 5e-4} | the STE has no settings of its own, so it gets the general ones |
| EWGS | δ from the Hessian with G = 3σ, updated every 10 epochs | fixed δ ∈ {1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1}; Hessian with G = max; Hessian updated every 5 epochs | EWGS Table 4 |
| DSQ | α starts at 0.2 and is learned | α starts at 0.1, 0.3 or 0.4 and is learned; α fixed at 0.1, 0.2, 0.3 or 0.4; α learning rate 1e-4 instead of 1e-5 | DSQ Section 4.1 and Table 4 |

A thoroughly tuned STE makes any win against it more convincing. Step 8 must build the switches in this table: fixed δ, the G variant and the update interval for EWGS; fixed α, the initial α and the α learning rate for DSQ.

**Two laptops.** The tuning is split: the STE and the staircase on laptop A (RTX 3050, 4 GB), EWGS and DSQ on laptop B (RTX 3050, 6 GB). To keep the comparison clean (see [SETUP.md](SETUP.md)):
- **Same setup:** both laptops use the same commit, software versions, data, validation split and float models, verified with `check_env.py`.
- **One machine per tuning:** each method's tuning stays on one machine.
- **One machine per seed:** all methods for a given seed run on the same laptop, so no comparison is made across machines. Every result file records the GPU.

**Cost.** 56 tuning runs, about 23 GPU hours at 60 epochs:
- STE 4.4 h and staircase 6.1 h (measured), about 10.5 h on laptop A;
- EWGS and DSQ about 6.1 h each (estimated), about 12.2 h on laptop B.

The split is rebalanced once laptop B's speed is measured.

**In the paper:**

> Every method receives the same tuning budget of 14 runs on the development seed: the learning rate first, then the method's own hyperparameters, with search ranges taken from the original papers' ablations. Table X lists all settings tried.

### 5. Fix the metric and the win rule before step 10

**Status:** the metric is agreed, implemented and tested. The win rule is still to confirm.

**Primary metric.** Test accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation.

**How re-estimation works.** It recomputes the BatchNorm running mean and variance with exact rounding, done the same way for every method (`models/quant_layers.reestimate_bn` with `datasets.get_bn_loader`):
- **Data:** the full training split (45k images) with the training augmentation. Each image's random crop and flip are fixed by a seed, so every pass gives the same statistics, on either laptop and with any number of workers.
- **Procedure:** one forward pass, with a plain average over batches.
- **When:** only for the evaluations that count: the final epoch, for tuning (validation) and for results (test). At most every 10 epochs for training curves. One pass takes about 6 s, so doing it every epoch would add 20–30% to each run.
- **On a copy of the model:** the function changes the model in place, and the trained statistics are still needed for the second number.

We also report the accuracy without re-estimation, which is what the papers report.

**Why re-estimation.** The staircase trains with the soft curve S, so its BatchNorm statistics do not match exact rounding. The MLP study had no BatchNorm, so this problem is new on ResNet-20.

**Why with augmentation.** During training, BatchNorm always normalized *augmented* batches: random crops add black padding borders, and flips change the statistics too. Every later layer adapted to that normalization. Statistics of clean images normalize differently, even for images the network has never seen. Re-estimation must therefore use the input distribution BatchNorm saw during training.

A control experiment shows this is not memorization. Float models, seeds 0–2, accuracy on one half of the validation set, 2026-11-09:

| BatchNorm statistics | Seed 0 / 1 / 2 | Mean |
|---|---|---|
| Original (from training) | 91.60 / 91.28 / 92.00 | 91.63 |
| Re-estimated on training images, no augmentation | 87.80 / 87.00 / 86.72 | 87.17 |
| Re-estimated on training images, augmented (our method) | 91.64 / 91.44 / 92.04 | 91.71 |
| Re-estimated on the other validation half, never seen, no augmentation | 87.80 / 86.88 / 86.72 | 87.13 |
| Re-estimated on the other validation half, never seen, augmented | 91.36 / 91.44 / 92.28 | 91.69 |

Clean images fail whether or not the model has seen them; augmented images work whether or not it has. Using validation images for re-estimation was only for this diagnostic: reported numbers always re-estimate on the training split.

With augmentation, re-estimation also helps a quantized model: on seed 0, an STE 4/4 model before any training goes from 86.96 to 89.36, against 85.18 without augmentation. `check_env.py` checks that re-estimation keeps the float model's accuracy unchanged.

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

**Problem.** Everything does not fit (see the budget below), and the abstract can only use results that are finished before Nov 16.

**Change.** Drop 3/3; neither paper reports it on CIFAR-10. Run step 10 in this order, so that the results the abstract needs come first:
1. ResNet-20 at 2/2 and 4/4: STE vs staircase on the 4 fresh seeds. This comes before steps 8–9, because it is the abstract's main result.
2. ResNet-20 at 2/2 and 4/4: EWGS and DSQ on the same 4 fresh seeds, once steps 8–9 are done.
3. Ablations at 2/2, on the 4 fresh seeds: staircase with slope-1 backward; weights only and activations only for both the STE and the staircase.
4. MobileNetV2 at 4/4: STE vs staircase, 3 seeds, using ResNet-20's tuned settings. After the abstract.
5. CIFAR-100, only if time is left. After the abstract.

## Compute budget

Totals are in GPU hours, for one job at a time. "Measured" uses the per-epoch times above; "estimated" assumes EWGS and DSQ cost about as much as the staircase, and that MobileNetV2 QAT costs about 1.5 times our earlier placeholder measurement.

| Item | As written | Revised | Basis |
|---|---|---|---|
| Budget pilot (30, 60, 120 epochs) | — | 2.6 h | measured |
| Step 7 tuning, STE + staircase (28 runs) | 25 h (full grid, 60 epochs) | 10.5 h | measured |
| EWGS + DSQ tuning (28 runs) | — | 12.2 h | estimated |
| Step 9 reproductions (400 epochs) | 10 h | 10.0 h | partly estimated |
| Float models, seeds 3–4 | — | 1.8 h | measured |
| Step 10, ResNet-20 main comparison | 19 h (4 methods × 3 bit widths × 4 seeds) | 12.9 h (2 bit widths) | partly estimated |
| Step 10 ablations | 11 h | 7.6 h | measured |
| MobileNetV2 (float + QAT) | 33 h (4 methods) | 17.4 h (STE vs staircase) | estimated |
| CIFAR-100 | 12 h | optional | estimated |
| **Total** | **~117 h** | **~75 h** | |

**Available:** about 220 hours on two laptops, assuming each trains about 10 hours on each of the 11 nights from Nov 9 to Nov 19. The revised plan uses about a third of that, which leaves room for reruns.

## Abstract deadline (Nov 16, not editable)

The abstract cannot be changed after submission. It may therefore only claim results that are finished by about Nov 14, and those claims must stay true in the final paper. The paper may contain more than the abstract, but must not contradict it.

| Tier | Results | GPU time | Ready | What the abstract can then say |
|---|---|---|---|---|
| Must have | Float seeds 3–4; budget pilot; STE and staircase tuning; STE vs staircase on fresh seeds 1–4 at 2/2 and 4/4 | ~21 h | Nov 11 morning | How the staircase compares with the STE on ResNet-20 / CIFAR-10 |
| Should have | EWGS and DSQ: tuning, step 9 reproductions, fresh seeds 1–4 at 2/2 and 4/4 | ~29 h | Nov 12 morning | How the staircase compares with EWGS and DSQ, with our implementations checked against the papers |
| If time allows | Ablations | ~8 h | Nov 13 morning | Why it works: forward curve vs slopes, weights vs activations |
| After the abstract | MobileNetV2, CIFAR-100 | ~17 h + optional | Nov 20 | Nothing; these go into the paper only |

The first two tiers take about 50 GPU hours, about 25 on each laptop: today plus three nights. The nights of Nov 13 and 14 stay free as buffer for reruns.

**Risks.**
- The tuning can only start once `train_qat.py` exists and has passed short test runs, and the budget pilot has fixed the number of epochs. If the tuning slips by a night, the must-have results move to Nov 12.
- EWGS and DSQ tuning can only start on laptop B once step 8 is done, by the evening of Nov 10.

### Rules for writing the abstract

1. Numbers come only from fresh seeds 1–4, given as mean ± std, never from the tuning seed (seed 0).
2. Claims are scoped to ResNet-20 on CIFAR-10, so later MobileNetV2 or CIFAR-100 results cannot contradict them.
3. If EWGS and DSQ are not finished by Nov 14, the abstract states no outcome against them. The paper can still include the comparison.
4. Claims about the mechanism (forward curve vs slopes, weights vs activations) appear only if the ablations finish in time.
5. Every number in the abstract comes from a saved JSON file in `results/`.

## Revised schedule

Times assume laptop B is about as fast as laptop A; the split is rebalanced after its benchmark.

| Date | Laptop A (4 GB) | Laptop B (6 GB) | Daytime work |
|---|---|---|---|
| Nov 9 | Budget pilot (~2.6 h), then STE + staircase tuning (~10.5 h, into the morning) | Setup with SETUP.md and `check_env.py`; float seeds 3–4 (~2 h) | Write and test `train_qat.py` |
| Nov 10 | Tuning finishes. Night: STE vs staircase, fresh seeds 1–2 (~3 h) | Night: STE vs staircase, fresh seeds 3–4 (~3 h), then EWGS + DSQ tuning (~12 h, into Nov 11). **Must-have results final** | Step 8: EWGS and DSQ with their tuning switches and CPU tests |
| Nov 11 | EWGS 1/1 reproduction (~5 h). Night: EWGS + DSQ on fresh seeds 1–2 (~3.5 h) | Tuning finishes; DSQ 2-bit reproduction (~5 h). Night: EWGS + DSQ on fresh seeds 3–4 (~3.5 h). **Should-have results final** | Read the tuning results, decide the paper's framing |
| Nov 12 | Night: ablations, seeds 1–2 (~4 h) | Night: ablations, seeds 3–4 (~4 h) | Draft the abstract |
| Nov 13–14 | Buffer for reruns; MobileNetV2 float training | Buffer for reruns | Abstract draft with all numbers |
| Nov 15–16 | MobileNetV2 QAT | MobileNetV2 QAT | Finalize and submit the abstract |
| Nov 16–20 | MobileNetV2, then CIFAR-100 if time is left | same | Paper writing |

## Decisions to confirm

1. **Agreed (2026-11-09):** the QAT length is chosen by the budget pilot and its rule (change 1); 400 epochs for step 9 only.
2. **Agreed (2026-11-09):** one-at-a-time tuning, with 2/2's settings reused at 4/4.
3. **Agreed (2026-11-09):** seed 0 for tuning; seeds 1–4 for results.
4. **Agreed (2026-11-09):** option C, the same tuning budget of 14 runs for every method, split between the two laptops (change 4).
5. **Partly agreed (2026-11-09):** the primary metric, with BatchNorm re-estimation on the seeded, augmented training split (implemented and tested). Still to confirm: the win rule.
6. The step 9 targets: EWGS 1/1 and DSQ Table 4; DSQ 1/1 optional.
7. **Agreed (2026-11-09):** 3/3 dropped. Still to confirm: the priority order of step 10.
8. The abstract's scope: ResNet-20 on CIFAR-10 only, with numbers from fresh seeds 1–4.
9. The order of the work by abstract tier: must have, should have, if time allows, after the abstract.

Once these are agreed, the README is updated in its plan table, rules, numbers to reproduce, compute budget and status.

## Sources

- **Timings:** measured on 2026-11-09 with a throwaway script (not committed), using the code at commit `80158ed`.
- **Float results:** [results/float/](results/float/) and [pretrained/README.md](pretrained/README.md).
- **EWGS evaluation and training settings:** the official code, `CIFAR10/train_quant.py` and `train_fp.py` (see "Reference code" in the README for the pinned version).
- **DSQ settings:** the DSQ paper (Gong et al., ICCV 2019), Section 4.1, Section 4.3 and Tables 3–4.
- **MLP study:** [results/torch_qat/study_claude/STUDY_LOG.md](../results/torch_qat/study_claude/STUDY_LOG.md).
