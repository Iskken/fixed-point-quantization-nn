# Plan review and revised plan (2026-11-09)

**Status (2026-11-09):** decisions 1–5 agreed; decisions 6–10 still open (see [Decisions to confirm](#decisions-to-confirm)). The [README](README.md) plan will be updated once all are agreed. Decision 5 added BatchNorm re-estimation to the step 4–6 code (`models/quant_layers.reestimate_bn`, `datasets.get_bn_loader`); nothing else in steps 4–6 changed.

**Deadline constraint.** The abstract is due on **Nov 16** and **cannot be edited** after submission, so the work is ordered by what the abstract needs (see [Abstract deadline](#abstract-deadline-nov-16-not-editable)).

## Summary

The core plan holds. ResNet-20 on CIFAR-10 is the anchor; we compare our pure staircase with the STE, EWGS and DSQ under one protocol, and reproduce published EWGS and DSQ numbers to show our implementations are right.

Measured timings show that the plan as written needs about 117 GPU hours. The changes below bring it to about 94 GPU hours, against about 220 available on our two laptops before Nov 20. That includes a 2.6-hour pilot that fixes the training length, and an equal tuning budget for all four methods, tuned separately at each bit width. The work is shared between two laptops (RTX 3050 with 4 GB and with 6 GB), which roughly halves the wall-clock time. The changes also fix three methodological gaps:
- the same seeds are used for tuning and for results;
- EWGS and DSQ get no hyperparameter search;
- the win criterion is not fixed in advance.

Because the abstract cannot be edited after Nov 16, about 69 of those 94 GPU hours (about 35 on each laptop) are scheduled before it, plus about 8 hours of ablations if time allows. The rest goes into the paper only.

## Where we are

- **Steps 1–3 done:** folder structure, CIFAR-10/100 download, and three float ResNet-20 models (seeds 0–2). Their test accuracy is 91.17 / 91.16 / 91.23 (see [pretrained/](pretrained/)), which matches the papers' float baselines: EWGS 91.4, DSQ 90.78.
- **Steps 4–6 done:** the shared quantizer, the STE, the pure staircase (`exact` or slope-1 backward, α schedule) and the sanity tests (29 passing, including BatchNorm re-estimation).
- **`train_qat.py` written and tested (2026-11-09)** for the STE and the staircase: the pilot, tuning, fresh-seed and ablation runs can start. It passes 10 CPU smoke tests (39 tests in total) and ran 1–3 epochs on the GPU. EWGS and DSQ plug in at step 8.
- **4/4 tuning done (2026-11-12, laptop B).** STE: lr 1e-2, lr_q 1e-5, weight decay 1e-4 → 91.84 (12 runs). Staircase: lr 1e-4, α_end 3, β 2 → 90.94 (15 runs). Both lrs are at the extended edge. The exploratory runs below started the same day.

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

**Laptop B (6 GB), measured 2026-11-09 with `train_qat.py` at 2/2, plugged in, no throttling:** 20 s (STE) and 30 s (staircase) of training per epoch, plus about 1 s and 2.7 s of validation; `check_env --bench` gives the same (114 and 169 ms per batch). A 60-epoch run there takes about 22 min (STE) and 34 min (staircase), with the re-estimations and the final evaluation. Laptop B is therefore about 1.3 times slower than the reference laptop in `check_env.py`, and the split between the laptops should be rebalanced accordingly.

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
- **Fallback, fixed now.** If even 60 → 120 fails the rule, N = 120. Every 60-epoch cost below then doubles, and for the abstract EWGS and DSQ move from "should have" to "if time allows". If 30 → 60 passes, N = 30 and every later step costs half as much.
- **Limitation.** There is one run per budget, so the pilot detects large budget effects, not small ones. The paper says so.

**Cost.** The pilot takes about 2.6 GPU hours: 210 epochs per method, at about 19 s (STE) and 26 s (staircase) per epoch. At N = 60, a run takes about 19 min (STE) or 26 min (staircase).

**In the paper** (N and the largest observed change are filled in from the pilot):

> We fixed the fine-tuning budget in advance with a rule applied on the development seed: the shortest budget in {30, 60, 120} epochs for which doubling it changed each method's validation accuracy, and the accuracy gap between methods, by less than 0.3 points. This threshold is about 2.5 standard deviations of the seed-to-seed variation of our full-precision models on the validation set (0.12 points), and below half of the method improvements reported on this benchmark (+0.9 for EWGS over STE at 1/1 [EWGS, Table 5]; +0.62 for DSQ's α mechanism at 2 bits [DSQ, Table 4]). To validate our EWGS and DSQ implementations, we additionally reproduce published results under the papers' original 400-epoch protocol.

### 2. Tune one setting at a time, separately at each bit width

**Problem.** The full grid for the staircase (3 learning rates × 3 α_end × 3 β = 27 runs per bit width) takes about 23 hours at 2/2 and 4/4, for the staircase alone. Reusing the settings chosen at 2/2 at 4/4 would be cheaper, but in the MLP study the setting that won at 8/4 tied at 6/4 and lost at 6/3: settings that work at one bit width may not work at another.

**Change.** α and β are tuned separately at 2/2 and at 4/4, and never reused from another bit width. Comparing the two also shows whether the best setting depends on the bit width. At each bit width, on the development seed (seed 0), one setting at a time:
1. **Learning rate:** lr ∈ {3e-4, 1e-3, 3e-3} at the reference setting α_end = 5, β = 2, the centre of the grid. 3 runs.
2. **α and β:** at the best learning rate from step 1, the 3 × 3 grid α_end ∈ {3, 5, 10} × β ∈ {1, 2, 3}. 8 new runs; the reference is reused from step 1.
3. **Learning-rate re-check:** at the best α and β from step 2, the two learning rates not yet tried there. 2 runs, or none if the reference wins step 2, because step 1 already covers it. In the MLP study the best α and β changed with the learning rate, so the learning rate found at the reference may not suit the chosen setting.

The search stops after step 3: if the re-check changes the learning rate, step 2 is not repeated. That is up to 13 runs per bit width, plus edge extensions. The other methods follow the same procedure with their own settings (change 4).

**Why α_end ∈ {3, 5, 10} and β ∈ {1, 2, 3}.**
- **Changed on 2026-11-10, after the budget pilot.** The grid was first α_end ∈ {2, 3, 5}, centred on the MLP optimum (α ≈ 2, β 2–3), with reference α_end = 2. On the development seed, at α_end = 2, the deployed (exact-rounding) staircase was 7.1–8.2 points below its own accuracy with the stair S, and 6.7–8.4 points below the STE, at all three budgets (`results/qat/pilot/decision.json`). The grid is therefore moved towards exact rounding. This was decided on seed 0 only, before any tuning run of the staircase, and without the test set; the paper reports the change and its reason.
- **Closer to exact rounding:** averaged over a cell, the grid ranges from 0.125 (α_end 3, β 1) to 0.031 (α_end 10, β 3) of a level away from rounding; the old grid ranged from 0.167 to 0.057, and float is 0.250.
- **α_end = 2 dropped:** the pilot measured the gap above; it is still run once as the lower edge value if α_end = 3 wins.
- **β = 1 kept as a control:** it gives the symmetric stair, which is the identity at α = 1.
- **Roughly log-spaced:** about ×1.7 then ×2, which suits a parameter that changes the slopes by multiplication.
- **Expected failures:** the larger α_end, the flatter the flat zones: α_end = 5 with β = 3 has slopes 0.075 and 5.6, α_end = 10 with β = 3 has 0.035 and 10.6. In the MLP study very flat zones with the exact backward froze the weights, so failures there are recorded as results and do not change the grid.
- **In the paper:** the MLP study is why the first grid was chosen, and the seed-0 pilot is why it moved; neither is evidence that these values work on ResNet-20.

**Ramp and β schedule, fixed in advance.**
- **α:** linear from 1 to α_end over the first half of training (30 of 60 epochs), then constant. It is updated every step, not every epoch, so it rises smoothly.
- **β:** fixed for the whole run, with no ramp.
- **Why neither is tuned:** in the MLP study, other ramp lengths and ramping β gave no gain (at 8/4, fixed β scored 0.2075 and ramped β 0.2082). With a small α_end, the stair changes little during the ramp, so its length matters little (this argument is weaker for the new grid, with α_end up to 10; the optional ramp-fraction check below covers it). Every extra tuned setting adds compute and selection noise on seed 0.
- **Optional, after the main results:** ramp fraction ∈ {0.25, 0.5, 1.0} at the chosen setting at 2/2, reported as a sensitivity check and not used for tuning.

**Edge rule, fixed before tuning.** It applies to every ordered setting of every method (values in change 4). If the best value is at an edge of its grid, one value beyond that edge is run:
- if it is not better, the previous best is kept: the optimum lies inside the grid;
- if it is better, it is used, the grid is not extended again, and we report that the optimum may lie beyond the grid.

Extensions count as tuned runs.

**Selection metric.** Validation accuracy after the last epoch, with exact rounding, after BatchNorm re-estimation (change 5). The chosen setting goes unchanged to fresh seeds 1–4 (change 3).

**Also logged per run:**
- the gap between staircase-forward and exact-rounding accuracy;
- the share of weights that change level per epoch;
- the full per-epoch curves;
- the run time.

**Cost.** Staircase: up to 26 runs, about 11.3 h at 60 epochs (5.6 h per bit width). Together with the STE on the same laptop (change 4), about 19.5 h. If the pilot picks a different budget, this scales in proportion.

### 3. Use separate seeds for tuning and for results

**Problem.** Our rule is "tune on development seeds, claim on fresh seeds", but float models exist only for seeds 0–2. Tuning on seed 0 and reporting on seeds 0–2 breaks the rule. Selection on the tuning seed also flatters the chosen settings: with up to 13 settings tried per bit width, the best one wins partly by noise. Fresh seeds remove that bias.

**Change.** Train float models for **seeds 3 and 4**. Tune on seed 0 and report all step 10 results on **seeds 1–4**. Each run uses the float model and QAT seed with the same number, so all methods for a seed start from the same float model and are compared seed by seed. Runs on seed 0 never evaluate the test set, and `train_qat.py` enforces this. Every reported number, including the step 9 reproductions, therefore comes from seeds 1–4.

**Cost.** About 1 hour per float model.

### 4. Give every method the same tuning budget

**Problem.** If only the STE and the staircase are tuned, a reviewer would call the baselines under-tuned. Tuning only the learning rate of EWGS and DSQ would still leave the staircase with more tuning than the others.

**Change.** Every method (STE, staircase, EWGS, DSQ) is tuned with the procedure of change 2, separately at 2/2 and at 4/4, on seed 0:
1. 3 runs over the learning rate {3e-4, 1e-3, 3e-3} at the method's reference setting;
2. 8 runs over the rest of a grid of 9 of its own settings, at the best learning rate;
3. up to 2 runs re-checking the learning rate at the best setting.

That is up to 13 runs per bit width and 26 per method, plus edge extensions. With the same tuning for every method, a difference on the fresh seeds cannot be explained by one method getting more of it.

| Method | Reference (run first) | Its 8 other settings | Source of the range |
|---|---|---|---|
| Staircase | α_end = 5, β = 2 | the other 8 of α_end ∈ {3, 5, 10} × β ∈ {1, 2, 3} | our MLP study, moved towards rounding after the seed-0 pilot (change 2) |
| STE | quantizer learning rate 1e-5, weight decay 1e-4 | the other 8 of quantizer learning rate {1e-6, 1e-5, 1e-4} × weight decay {0, 1e-4, 5e-4} | the STE has no settings of its own, so it gets two that every method has |
| EWGS | δ from the Hessian with G = 3σ, starting at δ = 0 and updated every 10 epochs | fixed δ ∈ {1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1}; Hessian with G = max; one more setting, still open (decision 10) | reference: EWGS Section 4.1 and the official code (`--update_every 10`); G = max and fixed δ 1e-5, 1e-3, 1e-1: EWGS Table 4; fixed δ 1e-2: EWGS Table 5; fixed δ 1e-4, 3e-3: ours |
| DSQ | α starts at 0.2 and is learned, with α learning rate 1e-5; l and u learned | α starts at 0.1, 0.3 or 0.4 and is learned; α fixed at 0.1, 0.2, 0.3 or 0.4; α learning rate 1e-4 at the reference | initial α 0.2: DSQ Section 4.1; α kept in (0, 0.5): DSQ Section 4.2.3; learned vs fixed α: DSQ Table 4; the other values: ours |

- **STE:** the extra tuning on shared settings is deliberate: it can only make a staircase win harder. The paper says so.
- **EWGS's 8th setting (open, decision 10):** δ is 0 until the first Hessian update at epoch 10 (the official code never updates at epoch 0), and with δ = 0 the EWGS backward is exactly the STE. So EWGS trains as the STE for its first 10 epochs: 2.5% of the paper's 400 epochs, but 1/6 of our 60. The 8th setting could address this. The candidates:
  - **A.** Hessian updated every 5 epochs: shortens the STE phase to 5 epochs, but also doubles the number of updates.
  - **B.** δ also computed from the Hessian before epoch 0, then every 10 epochs as in the paper: removes the STE phase and changes nothing else.
  - **C.** G = mean |g| from EWGS Table 4: keeps the paper's schedule, but the paper already shows it is worse (83.1 against 85.6), so the run is probably wasted.

  EWGS Table 4 does not vary the update interval, so A and B are our own variants. Whichever is chosen is only an alternative in the tuning grid: EWGS uses it only if it beats the paper's reference on seed-0 validation.
- **DSQ:** the paper gives no learning rate for α, so the reference uses the quantizer learning rate of every other method (1e-5).
- **Step 8** must build the switches in this table: fixed δ, the G variant and the update schedule (decision 10) for EWGS; fixed α, the initial α and the α learning rate for DSQ.

**Edge values** (rule in change 2). Yes/no choices (learned or fixed α, the Hessian variants) have no edge.

| Setting | Below the grid | Above the grid |
|---|---|---|
| Learning rate | 1e-4 | 1e-2 |
| α_end | 2 | 20 |
| β | 0.5 | 4 |
| STE quantizer learning rate | 1e-7 | 1e-3 |
| Weight decay | — (0 is the lowest) | 1e-3 |
| EWGS fixed δ | 1e-6 | 3e-1 |
| DSQ initial α | 0.05 | — (DSQ keeps α below 0.5) |

Extensions count as tuned runs, so budgets can differ by a few runs.

**Reported:** a table giving, for each method and bit width, every setting tried, the chosen one, and the number of runs including extensions.

**Two laptops.** The tuning is split: the STE and the staircase on laptop A (RTX 3050, 4 GB), EWGS and DSQ on laptop B (RTX 3050, 6 GB). To keep the comparison clean (see [SETUP.md](SETUP.md)):
- **Same setup:** both laptops use the same commit, software versions, data, validation split and float models, verified with `check_env.py`.
- **One machine per tuning:** each method's tuning stays on one machine, since tuning only compares settings of one method with each other.
- **One machine per seed:** all methods for a given seed run on the same laptop, so no comparison is made across machines. Every result file records the GPU.

**Cost.** Up to 104 tuning runs, about 42 GPU hours at 60 epochs:
- laptop A: STE 26 × 19 min = 8.2 h and staircase 26 × 26 min = 11.3 h (measured), about 19.5 h;
- laptop B: EWGS and DSQ about 11.3 h each (estimated, as the staircase), about 22.5 h.

That is about a day of continuous running on each laptop. The split is rebalanced once laptop B's speed is measured.

**In the paper:**

> Every method receives the same tuning budget on the development seed, separately for each bit width: the learning rate first, then a grid of nine settings of the method's own hyperparameters, then a re-check of the learning rate, with ranges based on the original papers' ablations where they exist. Table X lists all settings tried.

### 5. Fix the metric and the win rule before step 10

**Status:** agreed. The metric is implemented and tested.

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

On the seed-0 float model, the augmented re-estimate also reproduces the original statistics layer by layer: running means within 0.005 standard deviations and variances within 0.5% in all 19 BatchNorm2d layers, averaged over each layer's channels. The clean re-estimate is off by up to 0.10 standard deviations and 26% (measured 2026-11-09).

With augmentation, re-estimation also helps a quantized model: on seed 0, an STE 4/4 model before any training goes from 86.96 to 89.36, against 85.18 without augmentation. `check_env.py` checks that re-estimation keeps the float model's accuracy unchanged.

**Win rule,** per bit width, over the 4 fresh seeds. The one pre-registered comparison is the staircase against the STE. The staircase beats the STE if:
1. its mean test accuracy is higher;
2. a paired t-test gives p < 0.025 (0.05 with a Bonferroni correction for the 2 bit widths);
3. it wins on at least 3 of the 4 seeds.

Otherwise we report a tie or a loss as such. The comparisons with EWGS and DSQ are reported descriptively, with the same three numbers; a win against them is claimed only at p < 0.0083 (Bonferroni over all 6 comparisons). With 4 seeds a Wilcoxon test cannot reach p < 0.05, so the t-test is the formal test, and the per-seed results are reported alongside it. With 3 degrees of freedom the test has little power: gaps of 0.2–0.3 points will probably come out as ties, and are reported as such.

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

## Exploratory runs after the 4/4 tuning (2026-11-12)

**Status: added after the 4/4 tuning results were seen. Seed 0 (validation) only; the test set stays untouched.** These runs are not part of the pre-registered procedure (changes 2 and 4).

**Why.** The 4/4 tuning chose the STE at lr 1e-2 (91.84) and the staircase at lr 1e-4, α_end 3, β 2 (90.94).
- **The staircase's lr hit the lowest allowed value (1e-4),** as at 2/2. Every higher lr ended below its start: 87.58, 80.30 and 55.34 at 3e-4, 1e-3 and 3e-3 (α_end 5).
- **The curves show the pattern.**
  - At lr ≥ 3e-4, the staircase's training loss rises during the α ramp even while the lr falls (α_end 3, lr 3e-4: 0.019 at epoch 10, 0.044 at epoch 60), and validation falls with it.
  - At lr 1e-4 it fits the training set at once (loss 0.010 by epoch 5) but its quantized weights change level much less: 4% → 1% of the weights per epoch (epochs 1 → 60), against 11% → 7% for the STE at lr 3e-4. Its validation accuracy stays flat after epoch 30, while the STE's rises by 1.3 as its lr decays.

**Supervisor's comment.** Two points:
1. The lr is not scheduled with α. A low lr converges slowly; a high lr diverges as α grows and the problem becomes ill-conditioned.
2. The cosine lr schedule and the α ramp run at the same time and may interfere, so they should be studied separately.

A single lr cannot fix both, because inside S the steep slope is βα² times the flat one (18× at α 3, β 2). The runs below therefore test both directions: lowering the lr as α grows (C), removing either schedule (B), and the earlier follow-ups (A).

**The new option.** Both learning rates (network and quantizer) are set before every step to

  lr(t) = lr₀ · c(t) · (α_start / α(t))^p

- c(t) is the cosine (`--lr-schedule cosine`, the recipe) or 1 (`--lr-schedule constant`).
- p is `--lr-alpha-power`: 0 is the recipe; 1 is the inverse schedule, lr ∝ 1/α, which tracks the steepest slope of S.
- At the defaults the run is the recipe, unchanged. Earlier runs keep their names and configs; all 49 finished runs were checked.

**Runs.** ResNet-20, 4/4, seed 0, 120 epochs. Settings are α_end 3, β 2, lr_q 1e-5 and weight decay 1e-4 unless stated. Group `explore`, queued by `explore.py` in this order:

| # | Question | Change from the chosen setting | lr₀ | Compare with (tuning) |
|---|---|---|---|---|
| C1 | inverse × cosine | p = 1 | 3e-4 | 89.66 (same lr, recipe) |
| C2 | inverse × cosine | p = 1 | 1e-3 | 88.40, failed |
| B1 | remove the cosine | constant lr | 1e-4 | 90.94 |
| B2 | remove the α ramp | α fixed at 3 | 1e-4 | 90.94 |
| B3 | remove both | α fixed at 3, constant lr | 1e-4 | B1, B2 |
| A3 | are the exact slopes the cause? | slope-1 backward | 1e-3 | 88.40, failed |
| C3 | inverse only | p = 1, constant | 3e-4 | 89.66 |
| C4 | inverse only | p = 1, constant | 1e-3 | 88.40, failed |
| A1 | slower ramp | α ramps until the end | 1e-4 | 90.94 |
| A2 | slower ramp, higher lr | α ramps until the end | 3e-4 | 89.66 |
| A4 | lower lr (edge question) | — | 3e-5 | 90.94 |
| A4′ | the same extension for the STE | STE | 3e-2 | 91.84 |

**Reading the B runs.** They form a 2×2, {α ramp, α fixed} × {cosine, constant}:
- current run vs B2: what the ramp does under the cosine;
- B1 vs B3: what the ramp does under a constant lr;
- a clear difference between the two means the schedules interact.

B1 alone is not compared with the recipe: without the decay, the weights cannot settle at the end.

**Rules.**
- **Score:** validation after re-estimation, as in the tuning.
- **Promising:** a run must beat 90.94 by more than 0.3, because one run is noisy by about 0.2–0.3.
- **Follow-ups, only if something is promising:**
  - the same schedule at α_end 5, to see whether a larger α becomes usable;
  - the STE on the same lr trajectory, so a gain is not just the lr schedule;
  - the winner at 2/2.
- **Adopting a winner:** replacing the tuned staircase recipe for the fresh-seed runs is a joint decision, recorded here as a change made after the tuning. The STE then gets the matched schedule too.

**Prior evidence.** The MLP study tried an inverse schedule (p = 1 and 0.5) and several ramp lengths, with no gain that held on fresh seeds (results/torch_qat/study_claude/STUDY_LOG.md).

**Cost.** 11 staircase runs at about 67 min and 1 STE run at about 44 min on laptop B: about 13 h.

## Compute budget

Totals are in GPU hours, for one job at a time. "Measured" uses the per-epoch times above; "estimated" assumes EWGS and DSQ cost about as much as the staircase, and that MobileNetV2 QAT costs about 1.5 times our earlier placeholder measurement.

| Item | As written | Revised | Basis |
|---|---|---|---|
| Budget pilot (30, 60, 120 epochs) | — | 2.6 h | measured |
| Step 7 tuning, STE + staircase (up to 52 runs) | 25 h (full grid, 60 epochs) | 19.5 h | measured |
| EWGS + DSQ tuning (up to 52 runs) | — | 22.5 h | estimated |
| Step 9 reproductions (400 epochs) | 10 h | 10.0 h | partly estimated |
| Float models, seeds 3–4 | — | 1.8 h | measured |
| Step 10, ResNet-20 main comparison | 19 h (4 methods × 3 bit widths × 4 seeds) | 12.9 h (2 bit widths) | partly estimated |
| Step 10 ablations | 11 h | 7.6 h | measured |
| MobileNetV2 (float + QAT) | 33 h (4 methods) | 17.4 h (STE vs staircase) | estimated |
| CIFAR-100 | 12 h | optional | estimated |
| **Total** | **~117 h** | **~94 h** | |

**Available:** about 220 hours on two laptops, assuming each trains about 10 hours on each of the 11 nights from Nov 9 to Nov 19. The revised plan uses less than half of that. The real constraint is the abstract: about 77 of the 94 hours fall before it (see below).

## Abstract deadline (Nov 16, not editable)

The abstract cannot be changed after submission. It may therefore only claim results that are finished by about Nov 14, and those claims must stay true in the final paper. The paper may contain more than the abstract, but must not contradict it.

| Tier | Results | GPU time | Ready | What the abstract can then say |
|---|---|---|---|---|
| Must have | Float seeds 3–4; budget pilot; STE and staircase tuning; STE vs staircase on fresh seeds 1–4 at 2/2 and 4/4 | ~30 h | Nov 11 morning | How the staircase compares with the STE on ResNet-20 / CIFAR-10 |
| Should have | EWGS and DSQ: tuning, step 9 reproductions, fresh seeds 1–4 at 2/2 and 4/4 | ~39 h | Nov 12 morning | How the staircase compares with EWGS and DSQ, with our implementations checked against the papers |
| If time allows | Ablations | ~8 h | Nov 12 evening | Why it works: forward curve vs slopes, weights vs activations |
| After the abstract | MobileNetV2, CIFAR-100 | ~17 h + optional | Nov 20 | Nothing; these go into the paper only |

The first two tiers take about 69 GPU hours, about 35 on each laptop. They fit before the abstract only if both laptops train almost continuously, day and night, from tonight until the morning of Nov 12. The nights of Nov 12, 13 and 14 stay free as buffer for reruns.

**Risks.**
- The tuning can only start once `train_qat.py` exists and has passed short test runs, and the budget pilot has fixed the number of epochs. If the tuning slips by a day, every tier moves by a day and the buffer shrinks to one night.
- EWGS and DSQ tuning can only start on laptop B once step 8 is done, by the evening of Nov 10.
- If the pilot picks N = 120, all tuning doubles and does not fit before the abstract with EWGS and DSQ (see the fallback in change 1).

### Rules for writing the abstract

1. Numbers come only from fresh seeds 1–4, given as mean ± std, never from the tuning seed (seed 0).
2. Claims are scoped to ResNet-20 on CIFAR-10, so later MobileNetV2 or CIFAR-100 results cannot contradict them.
3. If EWGS and DSQ are not finished by Nov 14, the abstract states no outcome against them. The paper can still include the comparison.
4. Claims about the mechanism (forward curve vs slopes, weights vs activations) appear only if the ablations finish in time.
5. Every number in the abstract comes from a saved JSON file in `results/`.

## Revised schedule

Times assume laptop B is about as fast as laptop A and that both train almost continuously; the split is rebalanced after laptop B's benchmark.

| Date | Laptop A (4 GB) | Laptop B (6 GB) | Daytime work |
|---|---|---|---|
| Nov 9 | Budget pilot (~2.6 h), then STE + staircase tuning at 2/2 and 4/4 (~19.5 h, until the evening of Nov 10) | Setup with SETUP.md and `check_env.py`; float seeds 3–4 (~2 h) | Write and test `train_qat.py` |
| Nov 10 | Tuning finishes. Night: STE vs staircase, fresh seeds 1–2 (~3 h) | Evening: STE vs staircase, fresh seeds 3–4 (~3 h), then EWGS + DSQ tuning (~22.5 h, until the evening of Nov 11). **Must-have results final** | Step 8: EWGS and DSQ with their tuning switches and CPU tests |
| Nov 11 | EWGS 1/1 reproduction (~5 h). Night: EWGS + DSQ on fresh seeds 1–2 (~3.5 h), then ablations, seeds 1–2 (~4 h) | Tuning finishes in the evening. Night: DSQ 2-bit reproduction (~5 h), then EWGS + DSQ on fresh seeds 3–4 (~3.5 h). **Should-have results final** | Read the STE vs staircase results, decide the paper's framing |
| Nov 12 | Buffer for reruns | Ablations, seeds 3–4 (~4 h); then buffer | Draft the abstract |
| Nov 13–14 | Buffer for reruns; MobileNetV2 float training | Buffer for reruns | Abstract draft with all numbers |
| Nov 15–16 | MobileNetV2 QAT | MobileNetV2 QAT | Finalize and submit the abstract |
| Nov 16–20 | MobileNetV2, then CIFAR-100 if time is left | same | Paper writing |

## Decisions to confirm

1. **Agreed (2026-11-09):** every method is fine-tuned for the same N epochs, fixed by the budget pilot's rule on seed 0 (change 1); only the step 9 reproductions use 400 epochs.
2. **Agreed (2026-11-09), grid changed 2026-11-10:** tuning goes one setting at a time (learning rate, then α_end ∈ {3, 5, 10} × β ∈ {1, 2, 3}, then a learning-rate re-check), separately at 2/2 and 4/4, with a fixed α ramp, fixed β and the edge rule (change 2). The grid was α_end ∈ {2, 3, 5} with reference α_end = 2; after the pilot it moved towards exact rounding, with reference α_end = 5 and edge values 2 and 20. Both of us still need to confirm the change.
3. **Agreed (2026-11-09):** seed 0 is used for tuning only, and all results come from fresh seeds 1–4, each with its own float model (change 3).
4. **Agreed (2026-11-09):** every method gets the same tuning procedure and budget, up to 13 runs per bit width, split between the two laptops (change 4).
5. **Agreed (2026-11-09):** the primary metric is test accuracy after the last epoch with exact rounding, after BatchNorm re-estimation on the seeded, augmented training split, and the one pre-registered win test is staircase vs STE (change 5).
6. **Open:** the step 9 targets: EWGS 1/1 and DSQ Table 4; DSQ 1/1 optional.
7. **Open:** dropping 3/3, and the priority order of step 10.
8. **Open:** the abstract's scope: ResNet-20 on CIFAR-10 only, with numbers from fresh seeds 1–4.
9. **Open:** the order of the work by abstract tier: must have, should have, if time allows, after the abstract.
10. **Open:** EWGS's 8th tuning setting (change 4): Hessian every 5 epochs (A), an extra Hessian update before epoch 0 (B), or G = mean |g| (C).

Once these are agreed, the README is updated in its plan table, rules, numbers to reproduce, compute budget and status.

## Sources

- **Timings:** measured on 2026-11-09 with a throwaway script (not committed), using the code at commit `80158ed`.
- **Float results:** [results/float/](results/float/) and [pretrained/README.md](pretrained/README.md).
- **EWGS evaluation and training settings:** the official code, `CIFAR10/train_quant.py` and `train_fp.py` (see "Reference code" in the README for the pinned version).
- **EWGS scaling factor δ:** the EWGS paper (Lee et al., CVPR 2021), Section 4.1 (δ starts at 0 and is updated every 10 epochs on CIFAR-10) and Tables 4–5; the official code, `CIFAR10/train_quant.py` (`--update_every 10`, no update at epoch 0) and `CIFAR10/utils.py` (`update_grad_scales`: 3 batches, G = 3σ).
- **DSQ settings:** the DSQ paper (Gong et al., ICCV 2019), Sections 4.1, 4.2.3 and 4.3, and Tables 3–4.
- **MLP study:** [results/torch_qat/study_claude/STUDY_LOG.md](../results/torch_qat/study_claude/STUDY_LOG.md).
