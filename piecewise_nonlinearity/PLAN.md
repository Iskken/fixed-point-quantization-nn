# Plan review and revised plan (2026-11-09)

**Status: proposed.** The [README](README.md) plan will be updated once we agree on the changes below. None of them affect the code of steps 4–6 (shared quantizer, STE, staircase, tests).

**Deadline constraint.** The abstract is due on **Nov 16** and **cannot be edited** after submission, so the work is ordered by what the abstract needs (see [Abstract deadline](#abstract-deadline-nov-16-not-editable)).

## Summary

The core plan holds. ResNet-20 on CIFAR-10 is the anchor; we compare our pure staircase with the STE, EWGS and DSQ under one protocol, and reproduce published EWGS and DSQ numbers to show our implementations are right.

Measured timings show that the plan as written needs about 117 GPU hours, against about 110 available before Nov 20. The changes below bring it to about 66 hours, including a 2.6-hour pilot that fixes the training length. They also fix three methodological gaps:
- the same seeds are used for tuning and for results;
- EWGS and DSQ get no hyperparameter search;
- the win criterion is not fixed in advance.

Because the abstract cannot be edited after Nov 16, about 41 of those 66 hours are scheduled before it. The rest goes into the paper only.

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
  2. At the staircase's best learning rate, run the other 8 combinations of α_end ∈ {1.5, 2, 3} × β ∈ {1, 2, 3}.
- **At 4/4:** tune only the learning rate for both methods, reusing the α and β chosen at 2/2.

The 4/4 rule is fixed now, before any results.

**Selection metric.** Validation accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation.

**Also logged per run:**
- the gap between staircase-forward and exact-rounding accuracy;
- the share of weights that change level per epoch;
- the run time.

**Cost.** About 8 hours at 60 epochs: 5.7 h at 2/2 and 2.3 h at 4/4. If the pilot picks a different budget, this scales in proportion.

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
| Step 7 tuning | 25 h (full grid, 60 epochs) | 8.0 h | measured |
| EWGS / DSQ learning-rate tuning | — | 5.2 h | estimated |
| Step 9 reproductions (400 epochs) | 10 h | 10.0 h | partly estimated |
| Float models, seeds 3–4 | — | 1.8 h | measured |
| Step 10, ResNet-20 main comparison | 19 h (4 methods × 3 bit widths × 4 seeds) | 12.9 h (2 bit widths) | partly estimated |
| Step 10 ablations | 11 h | 7.6 h | measured |
| MobileNetV2 (float + QAT) | 33 h (4 methods) | 17.4 h (STE vs staircase) | estimated |
| CIFAR-100 | 12 h | optional | estimated |
| **Total** | **~117 h** | **~66 h** | |

**Available:** about 110 hours. That assumes the laptop trains about 10 hours on each of the 11 nights from Nov 9 to Nov 19. The revised plan leaves room for reruns.

## Abstract deadline (Nov 16, not editable)

The abstract cannot be changed after submission. It may therefore only claim results that are finished by about Nov 14, and those claims must stay true in the final paper. The paper may contain more than the abstract, but must not contradict it.

| Tier | Results | GPU time | Ready | What the abstract can then say |
|---|---|---|---|---|
| Must have | Float seeds 3–4; budget pilot; step 7 tuning; STE vs staircase on fresh seeds 1–4 at 2/2 and 4/4 | ~18 h | Nov 11 morning | How the staircase compares with the STE on ResNet-20 / CIFAR-10 |
| Should have | EWGS and DSQ: learning-rate tuning, step 9 reproductions, fresh seeds 1–4 at 2/2 and 4/4 | ~22 h | Nov 13 | How the staircase compares with EWGS and DSQ, with our implementations checked against the papers |
| If time allows | Ablations | ~8 h | Nov 14 morning | Why it works: forward curve vs slopes, weights vs activations |
| After the abstract | MobileNetV2, CIFAR-100 | ~17 h + optional | Nov 20 | Nothing; these go into the paper only |

The first two tiers take about 41 GPU hours: today plus four nights. The nights of Nov 13 and 14 stay free as buffer for reruns.

**Risk.** The tuning can only start once `train_qat.py` exists and has passed short test runs, and the budget pilot has fixed the number of epochs. If the tuning slips by a night, the must-have results move to Nov 12 and only one buffer night remains.

### Rules for writing the abstract

1. Numbers come only from fresh seeds 1–4, given as mean ± std, never from the tuning seed (seed 0).
2. Claims are scoped to ResNet-20 on CIFAR-10, so later MobileNetV2 or CIFAR-100 results cannot contradict them.
3. If EWGS and DSQ are not finished by Nov 14, the abstract states no outcome against them. The paper can still include the comparison.
4. Claims about the mechanism (forward curve vs slopes, weights vs activations) appear only if the ablations finish in time.
5. Every number in the abstract comes from a saved JSON file in `results/`.

## Revised schedule

| Date | Day | Night (GPU) |
|---|---|---|
| Nov 9 | Float seeds 3–4 (~2 h on the GPU); write and test `train_qat.py`; budget pilot in the afternoon (~2.6 h) | Step 7 tuning (~8 h at 60 epochs) |
| Nov 10 | Step 8: EWGS and DSQ with CPU tests, split between the two of us | STE vs staircase on fresh seeds (~6 h): **must-have results final** |
| Nov 11 | Short GPU check of EWGS and DSQ | EWGS/DSQ learning-rate tuning (~5 h) + EWGS 1/1 reproduction (~5 h) |
| Nov 12 | Draft the abstract from the must-have results | EWGS/DSQ on fresh seeds (~7 h) + DSQ 2-bit reproduction (~5 h): **should-have results final** (may run into the morning) |
| Nov 13–14 | Abstract draft with all numbers | Buffer for reruns, then ablations (~8 h) |
| Nov 15–16 | Finalize and submit the abstract | MobileNetV2 float training and QAT start |
| Nov 16–20 | Paper writing | MobileNetV2, then CIFAR-100 if time is left |

## Decisions to confirm

1. **Agreed (2026-11-09):** the QAT length is chosen by the budget pilot and its rule (change 1); 400 epochs for step 9 only.
2. One-at-a-time tuning, with 2/2's α and β reused at 4/4.
3. Seed 0 for tuning; seeds 1–4 for results.
4. The same learning-rate search for EWGS and DSQ.
5. The primary metric (with BatchNorm re-estimation) and the win rule.
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
