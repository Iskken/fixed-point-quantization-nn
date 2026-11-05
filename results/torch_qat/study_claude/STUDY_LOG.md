# Stair vs hard STE study: decision log

Started 2026-10-29 20:59 CET. Goal: find a setting where the stair (exact slopes or slope 1) beats hard
STE, under a fair protocol. Limit: 15 hours (until about 12:00 on 2026-10-30).

## Protocol (agreed before starting)

- ReLU MLP [4,32,32,16,1], Start B (best-validation float checkpoint of the lr 0.01 / 100k float run),
  everything quantized in training and evaluation, full batch, Adam (default betas unless stated),
  float64, batched GPU trainer (`src/models/torch_qat_batched.py`, parity-tested against `train_qat`).
- Dev seeds 0-2 for all tuning; selection on exact validation MSE only. Seeds 3-4 only for the 5-seed
  float-continuation control. Fresh seeds 5-14 (float starts made 2026-10-29) only for the final
  confirmation.
- Failed run = final validation MSE worse than PTQ of its start. Abort at epoch 60 if validation is
  above max(2, 2 x PTQ of the start). Cells with a failed seed rank last.
- Win = on fresh seeds 5-14: lower mean validation, lower mean test with a significant paired
  per-seed difference (t-test and Wilcoxon, p < 0.05), no more failed seeds than hard STE, and a
  neighbouring setting that also beats or ties hard STE.
- Every run saved with `save_run` (all curves, states at init / best / every 10k epochs, full task in
  meta) under `raw/`; task lists under `tasks/`; summary tables as `summary_*.md`.

## Facts established before the study

- PTQ of the start (validation, seeds 0-4): float 0.206; 8/4 0.338; 7/4 0.341; 6/4 0.460; 6/3 0.667;
  5/3 0.718; 4/2 1.418. The lsb, not the range, drives the loss.
- Input quantization alone limits any predictor to MSE 0.027 at 8/4 and 6/4, 0.094 at 6/3 and 5/3,
  0.256 at 4/2 (Monte Carlo, conditional variance of the target given quantized inputs).
- GPU vs CPU: hard STE bitwise identical; stair runs drift apart after ~400-1600 epochs (Adam amplifies
  ~1e-17 differences in the weight gradients), so single stair runs carry ~0.02 numerical noise.
- 8/4 before the study (seeds 0-2, Adam): hard STE best lr 0.03 (val 0.177, test 0.187); slope 1 best
  lr 0.05 alpha 30 (val 0.181, test 0.193); exact plateaus at val 0.24-0.27 for lr 3e-6..1e-4,
  alpha 10..100. Slope 1 at lr 0.07 and hard STE at lr 0.05 fail on some seeds.

## Log

- 20:59 Float starts for seeds 5-14 launched (CPU). Done by 21:12; best float validation 0.18-0.25,
  similar to seeds 0-4 (0.19-0.23).
- Phase 0: trainer extended (per-run ramp, beta, gradient floor, Adam betas, patience, abort threshold;
  float-continuation mode; diagnostics). Parity test 28/28 pass (`experiments/test_torch_qat_batched.py`).
  Study runner `run_study.py`, task lists `study_tasks.py`, tables `study_report.py`.
- Phase 1 + hard STE grids at 6/4, 6/3, 4/2 launched (105 runs, `tasks/phase1+phase2_hard.json`).
  4/2 stair grid (96 runs) queued after it (it does not depend on the 6-bit format choice).
- 21:31 Phase 1 + hard STE grids done (105 runs). Tables: `summary_hard_q*f*.md`, `summary_phase1_8bit.md`,
  `summary_float_control.md`.
  - Hard STE best lr per format (dev val / test): 8/4 lr 0.03 0.177/0.187; 6/4 lr 0.02 0.180/0.185;
    6/3 lr 0.02 0.229/0.265; 4/2 lr 0.01 0.288/0.335. Higher lrs fail on some seeds at every format.
  - Format decision: by the agreed rule 6 bits = 6/4 (better hard STE val). But 6/4 hard STE is already
    at 8/4 level (no headroom), while 6/3 hard STE (0.229) sits far above its input floor (0.094) and the
    float level (0.206): the most room for a better quantizer. Decision: run the stair grid at BOTH 6/4
    (agreed) and 6/3 (headroom). 4/2 hard STE is close to its floor (0.288 vs 0.256).
  - Float-continuation control (5 seeds, Adam lr 0.03): val 0.202 / test 0.204 vs float start 0.206:
    extra training plus selection explains almost none of hard STE's val gain (to 0.177).
  - Repeat noise: the diagnostics rerun of slope 1 lr 0.05 alpha 30 (identical settings, different GPU
    batch) gave val 0.221 vs 0.181 originally (seed 2 test 0.197 -> 0.336). Hard STE reruns are bitwise
    identical; exact moved 0.006. lr 0.05 is at slope 1's stability edge. Consequence: leading cells get
    dev seeds 3-4 added before any selection.
  - mask_saturation off: no effect beyond noise (exact 0.244 vs 0.243; slope 1 0.206 vs 0.181/0.221).
  - Exact at alpha 3 / 5 (lr 1e-5..1e-4): val 0.247-0.305, alpha 3 fails on 1-2 seeds. No rescue.
  - Diagnostics (seeds 0-2): share of quantized params changing grid value per epoch: hard STE 2-3%,
    slope 1 1-2.5%, exact (lr 3e-5) 0.01-0.02%, after the ramp 0.001%: exact's quantized weights are
    nearly frozen at the lrs where it is stable. Pre-activation saturation: hard STE 3-5%, slope 1 9-12%,
    exact 0%. Stair-forward vs exact val gap at the best epoch <= 0.006 for all.
  - Generalization gap (final exact val - train): hard STE +0.053, slope 1 +0.077, exact +0.113,
    float continuation +0.053. Exact fits the training set as well as hard STE but generalizes worse
    (hypothesis, untested: with frozen grid assignments it fits training-set rounding details).
- 22:07 4/2 stair grid done (96 runs, `summary_q4f2.md`). Hard STE best lr 0.01: val 0.270 / 0.329 / 0.265
  (seeds 0-2). Best stable slope 1 (lr 0.02 alpha 100): 0.376 / 0.404 / 0.361, worse on every seed.
  Exact best (lr 3e-4 alpha 10, at the grid edge): val 0.542; extended to lr 1e-3 / 3e-3
  (`ext_q4f2_exact`, queued). Slope 1 at lr 0.03-0.05 matches hard STE where it survives (seed 0: 0.265 vs
  hard STE 0.256 at lr 0.05) but seed 1 is aborted at epoch 60 in every such cell; hard STE at lr 0.03-0.05
  fails on seed 2 the same way.
- New idea (justified by the evidence): at every precision the best lrs sit at the edge where runs blow up
  within the first ~60 epochs, and the stair matched hard STE only in that high-lr regime. A short linear
  lr warm-up (500 epochs, applied identically to every method) may make that regime usable. Not the same as
  the E3c warm-up (10k epochs from 10% lr), which hurt. Implemented as per-run `warmup` in the batched
  trainer (parity test 30/30) and run_study.
- GPU job queue added (`experiments/torch_qat/study_queue.sh`, reads `queue.txt`).
- 23:16 6/4 + 6/3 stair grids done (192 runs, `summary_q6f4.md`, `summary_q6f3.md`). Hard STE stays best at both:
  6/4 hard lr 0.02 val 0.180 vs best stair exact lr 1e-4 a10 0.248 / slope 1 lr 0.01 a30 0.254 (slope 1 lr 0.05
  fails 1 of 3 seeds, 0.214 on the other two); 6/3 hard lr 0.02 0.229 vs best slope 1 lr 0.03 a100 0.250, exact 0.343.
- Queue bug: study_queue.sh waited on `pgrep -f run_study`, which matched the shell that launched it, so the GPU sat
  idle 23:16-00:26. Fixed (pattern anchored on the python command); ext_q4f2_exact started 00:26.
- 00:34 ext_q4f2_exact done: exact at 4/2 improves with lr (lr 3e-3 a5 val 0.423, lr 1e-3 a5 0.460) but stays far
  behind hard STE (0.288); still at the grid edge. `warmup` phase launched (129 runs, `tasks/warmup.json`): warm-up
  500 for hard STE and slope 1 (a30/a100) at lr 0.03/0.05/0.07 on all four formats; 4/2 exact at lr 1e-2/3e-2 (a3/a5)
  and warm-up 500 at lr 3e-3/1e-2/3e-2 (a5).
- 01:14 warm-up phase done (129 runs). Negative result. Warm-up 500 does not help hard STE (8/4 best 0.196 vs 0.177
  without; 6/3 lr 0.07 0.229 = same as without) and clearly hurts slope 1 (8/4 0.26-0.29 vs 0.181; 6/3 0.32+ vs 0.250).
  Reading: slope 1 profits from the full lr while alpha is still near 1 (float-like phase); the high-lr instability is
  not fixed by a gentle start. 4/2 exact: lower alpha + higher lr helps (lr 1e-2 a3 0.362; lr 3e-2 a3 0.353 but 1
  failed seed), still far from hard STE 0.288. Warm-up dropped from further phases.
- 01:30 Phase 3 launched (`tasks/phase3.json`, candidates in `phase3.candidates.json`): ramps 5k/20k/40k and inverse lr
  (p 1, 0.5) on 2 slope-1 + 2 exact cells per format (16 candidates, 336 runs).
- 03:10 Phase 3 half done (8/4, 6/4: 176 runs). 8/4 lead: slope 1 lr0 0.05 a30 inverse p0.5, val 0.1805 / test 0.1802
  (seeds 0-2; per-seed val 0.167/0.196/0.179) vs hard STE lr 0.03 0.1774 / 0.1867 (0.167/0.176/0.189). Within noise on
  val, lower on test. Fragile: every neighbour at lr 0.05 (a10 inverse, p1, ramps 5k/20k/40k) fails seed 2 (val ~0.37).
  Other ramp/inverse cells: no gain (exact best 0.233 at 8/4 ramp 5k, 0.239 at 6/4 ramp 5k; slope 1 at 6/4 0.250).
- Fairness: trainer + runner now let hard STE follow a stair's lr trajectory (schedule "inverse" with alpha_end /
  transition / power; option judge_after_ramp). Parity test 30/30; lr trajectory checked to 1e-15.
  `lead_q8f4` queued (54 runs): lead + parent on seeds 3-4, 6 neighbours on seeds 0-4, hard STE lr 0.03 / 0.05 on the
  same trajectory (both judge modes) on seeds 0-4.
- 04:29 lead_q8f4 done. Lead NOT confirmed: on seed 3 the whole slope-1 lr 0.05 family blows up (val 0.645, same for
  constant lr, a30/a100 inverse, p 0.25/0.5/0.75), so the lead has 1 failed seed of 5 vs 0 for hard STE. Stable
  neighbour lr0 0.04: val 0.204 (5 seeds) vs hard STE 0.177. lr0 0.07 fails 4/5. Hard STE on the same lr trajectory:
  0.178 (stopping from epoch 0), 0.217 (judged after the ramp): the decreasing lr does not help hard STE either.
- Phase 3 at 6/3 / 4/2 (done 04:12): 6/3 slope 1 lr 0.03 a100 inverse p1 val 0.2307 / test 0.279 vs hard STE lr 0.02
  0.2292 / 0.265 (within noise on val, worse test); ramp 5k a100 0.2416 / 0.249. 4/2: best stair 0.361 vs hard 0.288.
- GPU idle 04:29-09:08 (queue empty, nothing scheduled overnight); ~3 h of budget left.
- 09:09 final batch `final_q6f3_curriculum` (51 runs): 6/3 close cells + hard STE lr 0.02 on seeds 3-4, hard STE on the
  inverse cell's lr trajectory (seeds 0-4); curriculum stair -> hard STE (lr 0.003 / 0.01) vs hard -> hard control at
  8/4 and 6/3, seeds 0-4.
- 09:13 final batch crashed after 11 runs: the 8/4 slope-1 stage-1 runs of seeds 0-2 live in relu_gpu/raw. Fixed
  (run_study falls back to the old sweep's file name); rerun as `final_q6f3_curriculum_rerun` at 10:54.
- 6/3 over 5 dev seeds: hard STE lr 0.02 val 0.223 / test 0.260. Slope 1 lr 0.03 a100 inverse p1 0.243 / 0.299
  (wins 1/5 seeds on val); ramp 5k 0.246 / 0.274 (wins 0/5 on val, p_t 0.04 against it). Hard STE on the inverse lr
  trajectory 0.224: no change. The 6/3 near-tie of seeds 0-2 does not hold. No cell qualifies for fresh seeds:
  stopping rule (b).
- 10:57 curriculum done (40 runs, seeds 0-4). Stage 2 = hard STE lr 0.003 / 0.01 from the best state of the stage-1 run;
  control = the same stage 2 from the best hard STE run. Paired curriculum - control (seeds whose stage 1 did not fail):
  8/4 val -0.008 (p_t 0.36 / 0.34), test -0.001 / -0.012 (p 0.95 / 0.61), seed 3 lost (its stage-1 run failed);
  6/3 val -0.004 / -0.003 (p 0.42 / 0.71), test +0.012 / +0.003 (p 0.34 / 0.82). No significant effect; test mixed.
- STOP (rule b): ideas 1-5 worked through (phase 5 exact variants dropped for time; exact never came within 0.06 of
  hard STE on validation). No stair setting is credibly ahead of hard STE on dev validation; no fresh-seed confirmation
  run. Fresh seeds 5-14 remain unused.

## Follow-up (user request 2026-10-30 ~11:00): make exact beat hard STE

Plan agreed: A diagnostics with an oscillation counter (wq_flipback: grid value A -> B -> A) for hard STE; C1 beta for
exact, fixed per run (0.3 / 3 / 10) and ramped 1 -> beta_end with alpha (3 / 10; alpha = 1 with beta != 1 is not the
identity, so the fixed variant starts on a distorted forward), alpha 2 / 3 / 5, lr 3e-4 / 1e-3 / 3e-3, 8/4 and 6/3;
C2 exact leftovers (grad floor 0.1 / 0.3, Adam beta2 0.99, patience 15k; hard STE patience 15k too); D hybrid (real
rounding forward, exact stair slope backward); E confirmation only if within noise. Gradient clipping and slope-1 work
dropped (user).
- 11:12-11:21 an older version of the task list (`extra_diag_beta`) ran by mistake (a cancelled command had already
  queued it); its 14 hard STE diagnostic runs (A) are kept, its stair batch was stopped before any run finished. The 3
  older 8/4 hard STE diag runs without the oscillation counter are in raw/superseded_diag/ (reruns bitwise identical).
- 11:21 `exact_beta_fixed` launched (192 runs: C1 fixed beta, C2). Trainer: `beta_ramp` option (parity test 32/32).
  `exact_beta_ramp` queued (108 runs).
- A result (hard STE, flip-back counter): 1-2.7% of quantized params change grid value per epoch; 6-10% of those
  changes return to the value of two steps before (A -> B -> A) before the best epoch, 13-18% after it (8/4, 6/3),
  10% at 4/2. Oscillation exists but is not dominant; it grows once validation stops improving.
- Hybrid implemented (method "hybrid": `hard_forward` in the batched stair op; forward = fixed_point_quantize, backward
  = exact stair slopes, masked as hard STE). Tests 36/36, incl. hybrid with slope-1 backward == hard STE bitwise over
  300 Adam epochs. `hybrid_b1` queued (96 runs: alpha 2/3/5/10 x lr 1e-3..3e-2, 8/4 and 6/3, beta 1). The beta part
  of D follows once C1 shows which beta (fixed / ramped) works.
- 12:45 `exact_beta_fixed` done (192 runs). beta unfreezes exact: share of quantized params changing grid value per
  epoch (2000 epochs before the best) 0.001% at beta 1 (lr 3e-5 a30) -> 0.13-0.23% at beta 0.3 a5 / 3 a2 / 10 a2,
  lr 3e-3 (hard STE 1.4%); val - train gap 0.113 -> 0.04-0.07 (hard STE 0.053). 8/4 best exact: beta 0.3 a5 lr 3e-3
  val 0.203 / test 0.236 (seeds 0.217 / 0.210 / 0.182), beta 3 a2 lr 3e-3 0.208 / 0.225 (0.210 / 0.213 / 0.200), vs
  previous best exact 0.233 and hard STE 0.177 (0.167 / 0.176 / 0.189): gap halved (0.056 -> 0.026), still behind on
  2 of 3 seeds. 6/3: best 0.277 (beta 0.3 a5 lr 3e-3) vs 0.343 before, hard STE 0.229. 19 of 56 beta cells fail
  (mostly beta 10 / a5 at high lr). The best cells are at the lr grid edge -> `exact_beta_ext` (84 runs) queued before
  hybrid_b1.
- C2 (beta 1, exact's best cells): grad floor 0.1 / 0.3, Adam beta2 0.99, patience 15k: no gain (8/4 0.233-0.240 vs
  0.243; 6/3 0.330-0.356 vs 0.343). Hard STE with patience 15k identical to patience 5k.
- 13:19 PAUSED on user request (SIGSTOP; queue pid 44625, training pid 122886, in `exact_beta_ramp`, 54 of 108 runs
  done, the rest at epoch ~8000). Resume: `kill -CONT 122886 44625`. Not yet run: `exact_beta_ext`, `hybrid_b1`.
- 14:23 resumed (kill -CONT).
- 14:36 `exact_beta_ramp` done (108 runs). Ramped beta (1 -> beta_end with alpha) is no better than fixed beta: 8/4 best
  beta 3 a2 lr 3e-3 0.2082 ramped vs 0.2075 fixed, beta 10 a2 0.2136 vs 0.2151; 6/3 best 0.277 ramped (beta 10 a2) vs
  0.277 fixed (beta 0.3 a5). Ramped fails slightly less at high lr (e.g. beta 10 a3 lr 3e-3: 0 vs 1 failed), otherwise
  within noise. Decision: fixed beta for the hybrid. `hybrid_beta` (72 runs: beta 0.3 / 3 x a 2 / 5 x lr 3e-3..3e-2)
  queued after hybrid_b1.
- 15:14 `exact_beta_ext` done (84 runs). FIRST EXACT LEAD: exact lr 1e-2, alpha 2, beta 3 (fixed): 8/4 val 0.1782 /
  test 0.1896 (seeds 0.173 / 0.179 / 0.183) vs hard STE lr 0.03 0.1774 / 0.1867 (0.167 / 0.176 / 0.189): tie within
  noise; 6/3 val 0.2177 / test 0.2550 (0.209 / 0.239 / 0.205) vs hard STE lr 0.02 0.2292 / 0.2650 (0.237 / 0.237 /
  0.214): lower on val on 2 of 3 seeds and on mean test. Neighbour beta 0.3 a5 lr 1e-2: 8/4 0.1903, 6/3 0.2224 (also
  below hard STE on 6/3 val). lr 3e-2 fails on most seeds; beta 0.3 / 0.1 at a10 worse.
- Queued (ahead of hybrid_beta): `exact_lead` (62 runs: lead + neighbour on dev seeds 3-4; neighbourhood lr 5e-3 / 2e-2,
  beta 2 / 5, alpha 1.5 on seeds 0-4; both formats) and `fresh_exact_lead` (40 runs: the lead vs best hard STE on fresh
  seeds 5-14, both formats). The lead was chosen on dev seeds 0-2 only, so the fresh seeds stay unused for selection;
  the fresh runs are queued before seeing seeds 3-4 to save time, and are read only after the dev check.
- 16:01 `hybrid_b1` done (96 runs, beta 1). 8/4 best hybrid lr 0.01 a3 val 0.1774 / test 0.1997 (= hard STE val, worse
  test), lr 0.03 a3 0.1798 / 0.1878; 6/3 best lr 0.03 a2 0.2206 / 0.2499 and lr 0.01 a3 0.2207 / 0.2547 vs hard STE
  0.2292 / 0.2650 (seeds 0-2). Hybrid cuts oscillation 4-6x (flip-back 0.03-0.04% / epoch vs hard STE 0.20%) without a
  clear MSE gain. alpha 10 bad (0.26+, fails at high lr).
- 16:31 `exact_lead` done (62 runs). Dev seeds 0-4, paired vs hard STE on val:
  8/4: lead (a2 beta 3) 0.1841 vs 0.1774 (+0.007, wins 1/5) -- slipped; neighbours a1.5 beta 3 0.1697 (-0.008, p 0.39,
  wins 4/5), a2 beta 2 0.1731 (-0.004, p 0.65, 3/5); beta 5, lr 5e-3 / 2e-2, a3 worse.
  6/3: lead 0.2128 vs 0.2233 (-0.011, p 0.34, wins 3/5; test -0.010, p 0.35); a2 beta 2 0.2133 (-0.010, p 0.19; test
  -0.013, p 0.07, 4/5); beta 0.3 a5 0.2162 (-0.007). Within noise, but a neighbourhood ties or beats hard STE.
  Fresh seeds for the lead running (`fresh_exact_lead`); `fresh_exact_lead2` (a2 beta 2 both formats, a1.5 beta 3 at
  8/4; 30 runs) queued after it.
- 16:59 fresh seeds 5-14 done (`fresh_exact_lead`, `fresh_exact_lead2`; 70 runs), paired vs hard STE (no failures):
  8/4 hard STE lr 0.03 val 0.1800 / test 0.1972.
    exact a2 beta 2 lr 1e-2: val 0.1668 (-0.0133, p_t 0.094, p_w 0.084, wins 7/10), test 0.1884 (-0.0087, CI
    [-0.0176, +0.0001], p_t 0.053, p_w 0.037, wins 8/10).
    exact a1.5 beta 3: val 0.1656 (-0.0145, p_t 0.069, 7/10), test 0.1885 (-0.0087, p_t 0.23, p_w 0.13, 8/10).
    exact a2 beta 3 (the 3-seed lead): val 0.1773 (-0.003, p 0.78), test 0.2002 (+0.003): tie.
  6/3 hard STE lr 0.02 val 0.2142 / test 0.2450. exact a2 beta 2: 0.2192 / 0.2508 (+0.005 / +0.006, 5/10); a2 beta 3:
    0.2201 / 0.2464 (+0.006, p_t 0.05 against exact, 3/10 on val). The 6/3 dev advantage was selection noise.
  Verdict so far: 8/4 misses the win definition narrowly (test t-test p 0.053 > 0.05); 6/3 no. Per the rule
  (promising but within noise -> more seeds): float starts for seeds 15-24 trained (CPU, same recipe as 5-14), and
  `fresh2_q8f4` (30 runs: both 8/4 candidates + hard STE lr 0.03 on seeds 15-24) queued. Pre-specified analysis: the
  20 fresh seeds 5-24 pooled, per candidate, same win definition.
- 17:35 `hybrid_beta` done (72 runs, dev seeds 0-2). Best: 8/4 hybrid lr 0.03 a5 beta 0.3 val 0.1707 / test 0.1782,
  lr 0.03 a2 beta 3 0.1753 / 0.1800 (hard STE 0.1774 / 0.1867); 6/3 lr 0.01 a5 beta 0.3 0.2146 / 0.2463, lr 0.03 a5
  beta 0.3 0.2154 / 0.2175 (hard STE 0.2292 / 0.2650). Within dev noise; not taken to fresh seeds (focus is exact).
- 17:46 `fresh2_q8f4` done (30 runs, seeds 15-24). Hard STE lr 0.03 FAILS seed 22 (final val 0.422 > PTQ start 0.371);
  exact cells fail none. Pooled fresh seeds 5-24 at 8/4, paired vs hard STE:
  exact a2 beta 2 lr 1e-2, protocol view (means and paired tests over seeds where neither failed, n = 19):
    val 0.1661 vs 0.1777, d -0.0116 CI [-0.0196, -0.0035], p_t 0.0074, p_w 0.0008, wins 15/19;
    test 0.1861 vs 0.1950, d -0.0089 CI [-0.0154, -0.0024], p_t 0.0098, p_w 0.0046, wins 16/19.
    With hard STE's failed seed kept in (n = 20): test d -0.0216, p_t 0.11 (outlier inflates the variance), p_w 0.0023,
    wins 17/20. Failed seeds: exact 0, hard STE 1.
  neighbour exact a1.5 beta 3 (n = 19): val d -0.0119 (p_t 0.013, p_w 0.011), test d -0.0083 (p_t 0.069, p_w 0.045).
  WIN DEFINITION MET for exact a2 beta 2 lr 1e-2 at 8/4 (lower val, lower test with p_t and p_w < 0.05 -- also after a
  Bonferroni factor 2 for two candidates -- no more failures, neighbour beats on val and ties/beats on test). Caveats:
  on seeds 5-14 alone the test t-test was 0.053; the cell was chosen out of ~150 exact cells on dev seeds; at 6/3 the
  same family does NOT beat hard STE on fresh seeds (+0.005). Stopping rule (a): robustness check `robust_q6f4` (same
  cell at 6/4 vs hard STE lr 0.02, fresh seeds 5-14, 20 runs) queued.
- 17:53 `robust_q6f4` done (20 runs). 6/4, fresh seeds 5-14: exact a2 beta 2 lr 1e-2 vs hard STE lr 0.02: hard STE
  fails 1 seed, exact 0; on the 9 common seeds val 0.1757 vs 0.1772 (d -0.0014, p_t 0.74, wins 5/9), test 0.1968 vs
  0.1924 (d +0.0043, p_t 0.33, wins 2/9): a tie. Robustness: the 8/4 win does not carry over to 6/4 (tie) or 6/3
  (slightly worse). Exact a2 beta 2 never failed on 30 fresh runs; hard STE failed 2 of 30 (8/4 seed 22, one 6/4 seed).
- FOLLOW-UP FINISHED (stopping rule a, robustness checked). Final report given in chat.
