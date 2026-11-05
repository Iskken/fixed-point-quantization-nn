import json
import os
import sys

"""
Task lists for the stair-vs-hard-STE study (run with run_study.py). Each phase writes
results/torch_qat/study/tasks/<phase>.json, so every launch can be reproduced exactly.

Dev seeds 0-2 for all tuning (selection on validation only), seeds 3-4 only for the 5-seed controls,
fresh seeds 5-14 only for the final confirmation.

python -m experiments.torch_qat.study_tasks <phase>
"""

TASK_DIR = "results/torch_qat/study/tasks"
DEV = (0, 1, 2)
HARD_LRS = (0.003, 0.01, 0.02, 0.03, 0.05)


def cells(seeds, **kw):
    return [{"seed": s, **kw} for s in seeds]


def phase1():
    """8/4: float-continuation control, diagnostics reruns, saturation mask off, exact at small alpha,
    and the full hard STE lr grid at 8/4 in the study dir (bitwise equal to the CPU runs)."""
    t = []
    for lr in (0.003, 0.03):
        t += cells(range(5), method="float", lr=lr)
    for lr in HARD_LRS:
        t += cells(range(5) if lr == 0.03 else DEV, method="hard_ste", lr=lr)
    t += cells(DEV, method="hard_ste", lr=0.03, diagnostics=True)
    t += cells(DEV, method="soft_ste", lr=0.05, alpha_end=30, diagnostics=True)
    t += cells(DEV, method="exact", lr=3e-5, alpha_end=30, diagnostics=True)
    t += cells(DEV, method="soft_ste", lr=0.05, alpha_end=30, mask_saturation=False)
    t += cells(DEV, method="exact", lr=3e-5, alpha_end=30, mask_saturation=False)
    for a in (3, 5):
        for lr in (1e-5, 3e-5, 1e-4):
            t += cells(DEV, method="exact", lr=lr, alpha_end=a)
    return t


def phase2_hard():
    """Lower precision, step 1: hard STE lr grid at 6/4, 6/3 and 4/2 (picks the 6-bit format)."""
    return [c for bits in ([6, 4], [6, 3], [4, 2]) for lr in HARD_LRS
            for c in cells(DEV, method="hard_ste", lr=lr, bits=bits)]


SLOPE1_LRS, SLOPE1_ALPHAS = (0.01, 0.02, 0.03, 0.05), (10, 30, 100)
EXACT_LRS, EXACT_ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4), (3, 5, 10, 30, 100)


def phase2_stair(bits):
    """Lower precision, step 2: slope-1 and exact grids at one bit width (ramp 10k)."""
    t = [c for lr in SLOPE1_LRS for a in SLOPE1_ALPHAS
         for c in cells(DEV, method="soft_ste", lr=lr, alpha_end=a, bits=bits)]
    t += [c for lr in EXACT_LRS for a in EXACT_ALPHAS
          for c in cells(DEV, method="exact", lr=lr, alpha_end=a, bits=bits)]
    return t


RAMPS = (5000, 20000, 40000)


def phase3(candidates, seeds=DEV):
    """
    Ramp length and decreasing lr on candidate cells (dicts with method, lr, alpha_end, bits).
    Ramps: 5k / 20k / 40k (10k exists). Decreasing lr = lr0 * (1/alpha)**p, p in {0.5, 1}:
      exact    -- alpha ~ 1 at the start is the identity, so lr0 is float-like and the end-of-ramp lr
                  brackets exact's best constant lr: p=1 lr0 {1e-3, 3e-3}; p=0.5 lr0 {3e-4, 1e-3}
      soft_ste -- lr above the best constant diverges early, so lr0 in {best, 0.6 x best}
    The patience scales with the ramp only through the stopping rule (judged after the ramp), as before.
    """
    t = []
    for c in candidates:
        base = {k: c[k] for k in ("method", "lr", "alpha_end", "bits")}
        for r in RAMPS:
            t += cells(seeds, **base, transition=r)
        if c["method"] == "exact":
            lr0s = {1.0: (1e-3, 3e-3), 0.5: (3e-4, 1e-3)}
        else:
            lr0s = {p: (c["lr"], round(0.6 * c["lr"], 6)) for p in (1.0, 0.5)}
        for p, lrs in lr0s.items():
            for lr0 in lrs:
                t += cells(seeds, **{**base, "lr": lr0}, schedule="inverse", power=p)
    return t


def from_file(fn):
    def make():
        spec = json.load(open(os.path.join(TASK_DIR, fn + ".candidates.json")))
        return fn_map[spec["kind"]](spec["candidates"], tuple(spec.get("seeds", DEV)))
    return make


def ext_q4f2_exact():
    """4/2: exact's best lr (3e-4) was at the grid edge -> extend upward at alpha 5 / 10 / 30."""
    return [c for lr in (1e-3, 3e-3) for a in (5, 10, 30)
            for c in cells(DEV, method="exact", lr=lr, alpha_end=a, bits=[4, 2])]


def warmup():
    """
    500-epoch linear lr warm-up, identical for every method, at the high lrs where runs blew up in the first
    ~60 epochs (and where slope 1 matched hard STE when it survived). All four formats. Plus 4/2 exact, whose
    best lr (3e-3) was again at the grid edge: extend to 1e-2 / 3e-2 at alpha 3 / 5, with and without warm-up.
    """
    t = []
    for bits in ([8, 4], [6, 4], [6, 3], [4, 2]):
        for lr in (0.03, 0.05, 0.07):
            t += cells(DEV, method="hard_ste", lr=lr, bits=bits, warmup=500)
            for a in (30, 100):
                t += cells(DEV, method="soft_ste", lr=lr, alpha_end=a, bits=bits, warmup=500)
    for lr in (1e-2, 3e-2):
        for a in (3, 5):
            t += cells(DEV, method="exact", lr=lr, alpha_end=a, bits=[4, 2])
    for lr in (3e-3, 1e-2, 3e-2):
        t += cells(DEV, method="exact", lr=lr, alpha_end=5, bits=[4, 2], warmup=500)
    return t


def lead_q8f4():
    """
    8/4 lead from phase 3: slope 1 lr0 0.05 alpha 30, inverse lr p 0.5 (val 0.1805 / test 0.1802 vs hard STE
    0.1774 / 0.1867 on seeds 0-2; within noise). Dev seeds 3-4 for it and its constant-lr parent, its
    neighbours (alpha, lr0, power) on seeds 0-4, and hard STE on the same lr trajectory (stopping rule from
    epoch 0, and after the ramp as for the stair) on seeds 0-4 for fairness.
    """
    lead = dict(method="soft_ste", lr=0.05, alpha_end=30, schedule="inverse", power=0.5)
    t = cells((3, 4), **lead) + cells((3, 4), method="soft_ste", lr=0.05, alpha_end=30)
    for nb in ({"alpha_end": 10}, {"alpha_end": 100}, {"lr": 0.04}, {"lr": 0.07}, {"power": 0.25}, {"power": 0.75}):
        t += cells(range(5), **{**lead, **nb})
    for lr in (0.03, 0.05):
        for jar in (False, True):
            t += cells(range(5), method="hard_ste", lr=lr, alpha_end=30, schedule="inverse", power=0.5,
                       judge_after_ramp=jar)
    return t


def final_q6f3_curriculum():
    """
    Last batch (time limit). 6/3: the two close slope-1 cells from phase 3 (lr 0.03 a100 inverse p1, val 0.2307;
    lr 0.03 a100 ramp 5k, 0.2416) and hard STE lr 0.02 on seeds 3-4, plus hard STE on the inverse cell's lr
    trajectory on seeds 0-4. Curriculum (phase 4) on seeds 0-4 at 8/4 and 6/3: stage 1 = best slope-1 run, stage 2 =
    hard STE at lr 0.003 / 0.01 from its best state; control = the same stage 2 from the best hard STE run.
    """
    from experiments.torch_qat.run_study import full, run_name
    seeds = range(5)
    inv = dict(method="soft_ste", lr=0.03, alpha_end=100, schedule="inverse", power=1.0, bits=[6, 3])
    t = cells((3, 4), **inv) + cells((3, 4), method="soft_ste", lr=0.03, alpha_end=100, transition=5000, bits=[6, 3])
    t += cells((3, 4), method="hard_ste", lr=0.02, bits=[6, 3])
    t += cells(seeds, method="hard_ste", lr=0.02, alpha_end=100, schedule="inverse", power=1.0, bits=[6, 3])
    stage1 = {(8, 4): [dict(method="soft_ste", lr=0.05, alpha_end=30), dict(method="hard_ste", lr=0.03)],
              (6, 3): [inv, dict(method="hard_ste", lr=0.02, bits=[6, 3])]}
    for bits, starts in stage1.items():
        for st in starts:
            for seed in seeds:
                prev = run_name(full({"seed": seed, **st}))
                for lr in (0.003, 0.01):
                    t.append({"seed": seed, "method": "hard_ste", "lr": lr, "bits": list(bits), "start": prev})
    return t


def extra_diag_beta():
    """
    Follow-up after the main study (user request, 2026-10-30).
    A: diagnostics with the oscillation counter (wq_flipback): hard STE at the best lr per format (and lr 0.05 at 8/4,
       which fails seed 2), slope 1 lr 0.05 a30 at 8/4 including the seed-3 blow-up, slope 1 6/3 lead.
    C: beta for exact. beta > 1 at small alpha keeps the flat zone flat (height lsb / (1 + beta alpha)) while the
       steep/flat slope ratio (beta alpha^2) stays moderate, so a higher stable lr may unfreeze the weights;
       beta 0.3 is the opposite check. Diagnostics on. Slope 1: beta 0.3 / 3 on its leading cells.
       Phase 5 leftovers: exact grad floor 0.1 / 0.3 and Adam beta2 0.99 at exact's best cell, patience 15k for the
       best cell of every method.
    """
    t = []
    for bits, lr, sd in (([8, 4], 0.03, DEV), ([6, 3], 0.02, range(5)), ([4, 2], 0.01, DEV), ([8, 4], 0.05, DEV)):
        t += cells(sd, method="hard_ste", lr=lr, bits=bits, diagnostics=True)
    t += cells(range(4), method="soft_ste", lr=0.05, alpha_end=30, diagnostics=True)
    t += cells(DEV, method="soft_ste", lr=0.03, alpha_end=100, schedule="inverse", power=1.0, bits=[6, 3], diagnostics=True)
    for bits in ([8, 4], [6, 3]):
        for beta in (0.3, 3, 10):
            for a in (2, 3, 5):
                for lr in (1e-4, 3e-4, 1e-3):
                    t += cells(DEV, method="exact", lr=lr, alpha_end=a, beta=beta, bits=bits, diagnostics=True)
    slope1 = [dict(lr=0.05, alpha_end=30, bits=[8, 4]), dict(lr=0.04, alpha_end=30, schedule="inverse", power=0.5, bits=[8, 4]),
              dict(lr=0.03, alpha_end=100, schedule="inverse", power=1.0, bits=[6, 3]),
              dict(lr=0.03, alpha_end=100, transition=5000, bits=[6, 3])]
    for c in slope1:
        for beta in (0.3, 3):
            t += cells(DEV, method="soft_ste", beta=beta, **c)
    exact_best = [dict(lr=3e-5, alpha_end=30, bits=[8, 4]), dict(lr=3e-5, alpha_end=100, bits=[6, 3])]
    for c in exact_best:
        for v in (dict(grad_floor=0.1), dict(grad_floor=0.3), dict(betas=[0.9, 0.99]), dict(patience=15000)):
            t += cells(DEV, method="exact", **c, **v)
    for c in (dict(method="soft_ste", lr=0.05, alpha_end=30, bits=[8, 4]), dict(method="hard_ste", lr=0.03, bits=[8, 4]),
              dict(method="soft_ste", lr=0.03, alpha_end=100, schedule="inverse", power=1.0, bits=[6, 3]),
              dict(method="hard_ste", lr=0.02, bits=[6, 3])):
        t += cells(DEV, **c, patience=15000)
    return t


EXACT_BETA_ALPHAS, EXACT_BETA_LRS = (2, 3, 5), (3e-4, 1e-3, 3e-3)


def exact_beta_fixed():
    """
    Updated plan (2026-10-30), focus on exact. C1: beta fixed for the whole run, beta 0.3 / 3 / 10 x alpha_end 2 / 3 / 5 x
    lr 3e-4 / 1e-3 / 3e-3 at 8/4 and 6/3, diagnostics on (share of quantized params changing / flipping back).
    beta > 1 at small alpha keeps the flat zone flat (rise lsb / (1 + beta alpha)) with a moderate steep / flat slope
    ratio (beta alpha^2), so a higher lr may be stable; beta 0.3 is the opposite check. Note: alpha = 1 with beta != 1 is
    not the identity, so these runs start on a distorted forward (the ramped-beta variant is exact_beta_ramp).
    C2: exact's best cells (beta 1) with grad floor 0.1 / 0.3, Adam beta2 0.99, patience 15k; hard STE patience 15k.
    """
    t = []
    for bits in ([8, 4], [6, 3]):
        for beta in (0.3, 3, 10):
            for a in EXACT_BETA_ALPHAS:
                for lr in EXACT_BETA_LRS:
                    t += cells(DEV, method="exact", lr=lr, alpha_end=a, beta=beta, bits=bits, diagnostics=True)
    for c in (dict(lr=3e-5, alpha_end=30, bits=[8, 4]), dict(lr=3e-5, alpha_end=100, bits=[6, 3])):
        for v in (dict(grad_floor=0.1), dict(grad_floor=0.3), dict(betas=[0.9, 0.99]), dict(patience=15000)):
            t += cells(DEV, method="exact", **c, **v)
    for c in (dict(lr=0.03, bits=[8, 4]), dict(lr=0.02, bits=[6, 3])):
        t += cells(DEV, method="hard_ste", **c, patience=15000)
    return t


def exact_beta_ramp():
    """C1, ramped variant: beta goes 1 -> beta_end (3 / 10) with the alpha ramp, so training starts at the identity
    (alpha = beta = 1) as for beta 1. Same alpha / lr / bits / seeds as exact_beta_fixed, diagnostics on."""
    return [c for bits in ([8, 4], [6, 3]) for beta in (3, 10) for a in EXACT_BETA_ALPHAS for lr in EXACT_BETA_LRS
            for c in cells(DEV, method="exact", lr=lr, alpha_end=a, beta=beta, beta_ramp=True, bits=bits, diagnostics=True)]


def hybrid_b1():
    """D, beta 1 part: hybrid = real rounding forward (as hard STE), exact stair slopes backward (1/alpha in the flat
    zone, alpha in the steep zone); alpha ramps 1 -> alpha_end, so it starts as hard STE. alpha_end 2 / 3 / 5 / 10 x
    lr 1e-3 / 3e-3 / 1e-2 / 3e-2 at 8/4 and 6/3, diagnostics on (oscillation counter vs hard STE)."""
    return [c for bits in ([8, 4], [6, 3]) for a in (2, 3, 5, 10) for lr in (1e-3, 3e-3, 1e-2, 3e-2)
            for c in cells(DEV, method="hybrid", lr=lr, alpha_end=a, bits=bits, diagnostics=True)]


def exact_beta_ext():
    """C1 extension: the best fixed-beta exact cells (8/4: beta 0.3 a5 val 0.203, beta 3 a2 0.208, beta 10 a2 0.215) sit at
    the top of the lr grid (3e-3) -> lr 1e-2 / 3e-2; beta 0.3 / 0.1 at alpha 10 (smaller steep / flat ratio than a5 at
    beta 1) at 3e-3 / 1e-2 / 3e-2. 8/4 and 6/3, diagnostics on."""
    combos = {(0.3, 5): (1e-2, 3e-2), (3, 2): (1e-2, 3e-2), (3, 3): (1e-2, 3e-2), (10, 2): (1e-2, 3e-2),
              (0.3, 10): (3e-3, 1e-2, 3e-2), (0.1, 10): (3e-3, 1e-2, 3e-2)}
    return [c for bits in ([8, 4], [6, 3]) for (beta, a), lrs in combos.items() for lr in lrs
            for c in cells(DEV, method="exact", lr=lr, alpha_end=a, beta=beta, bits=bits, diagnostics=True)]


def hybrid_beta():
    """D, beta part: fixed beta (ramped beta was no better for exact) at the two best exact shapes of C1, beta 0.3 a5 and
    beta 3 a2, plus the crossed pairs (beta 0.3 a2, beta 3 a5); lr 3e-3 / 1e-2 / 3e-2, 8/4 and 6/3, diagnostics on."""
    return [c for bits in ([8, 4], [6, 3]) for beta in (0.3, 3) for a in (2, 5) for lr in (3e-3, 1e-2, 3e-2)
            for c in cells(DEV, method="hybrid", lr=lr, alpha_end=a, beta=beta, bits=bits, diagnostics=True)]


def exact_lead():
    """
    Exact lead (exact_beta_ext): lr 1e-2 alpha 2 beta 3 (fixed). Seeds 0-2: 8/4 val 0.1782 vs hard STE 0.1774;
    6/3 val 0.2177 vs hard STE 0.2292. Dev seeds 3-4 for it and for the beta 0.3 a5 lr 1e-2 cell, and a neighbourhood
    on dev seeds 0-4 (lr 5e-3 / 2e-2, beta 2 / 5, alpha 1.5 / 3) at 8/4 and 6/3. Hard STE seeds 3-4 exist.
    """
    lead = dict(method="exact", lr=1e-2, alpha_end=2, beta=3, diagnostics=True)
    t = []
    for bits in ([8, 4], [6, 3]):
        t += cells((3, 4), **lead, bits=bits)
        t += cells((3, 4), **{**lead, "alpha_end": 5, "beta": 0.3}, bits=bits)
        t += cells((3, 4), **{**lead, "alpha_end": 3}, bits=bits)
        for nb in ({"lr": 5e-3}, {"lr": 2e-2}, {"beta": 2}, {"beta": 5}, {"alpha_end": 1.5}):
            t += cells(range(5), **{**lead, **nb}, bits=bits)
    return t


FRESH = tuple(range(5, 15))


def fresh_exact_lead():
    """Confirmation on fresh seeds 5-14 (pre-chosen on dev validation): exact lr 1e-2 a2 beta 3 vs the best hard STE of
    each format (8/4 lr 0.03, 6/3 lr 0.02)."""
    t = []
    for bits, hard_lr in (([8, 4], 0.03), ([6, 3], 0.02)):
        t += cells(FRESH, method="exact", lr=1e-2, alpha_end=2, beta=3, bits=bits, diagnostics=True)
        t += cells(FRESH, method="hard_ste", lr=hard_lr, bits=bits)
    return t


def fresh_exact_lead2():
    """Second fresh-seed candidates, chosen on dev seeds 0-4 after exact_lead: exact lr 1e-2 alpha 2 beta 2 (dev val 8/4
    0.1731, 6/3 0.2133: below hard STE at both formats) and, at 8/4, alpha 1.5 beta 3 (dev val 0.1697)."""
    t = []
    for bits in ([8, 4], [6, 3]):
        t += cells(FRESH, method="exact", lr=1e-2, alpha_end=2, beta=2, bits=bits, diagnostics=True)
    t += cells(FRESH, method="exact", lr=1e-2, alpha_end=1.5, beta=3, bits=[8, 4], diagnostics=True)
    return t


FRESH2 = tuple(range(15, 25))


def fresh2_q8f4():
    """Promising but within noise on fresh seeds 5-14 at 8/4 (exact a2 beta 2: test -0.0087, p_t 0.053, p_w 0.037, 8/10;
    a1.5 beta 3: -0.0087, p_t 0.23) -> 10 more seeds (15-24, float starts made 2026-10-30) for both and hard STE lr 0.03."""
    return (cells(FRESH2, method="exact", lr=1e-2, alpha_end=2, beta=2, diagnostics=True)
            + cells(FRESH2, method="exact", lr=1e-2, alpha_end=1.5, beta=3, diagnostics=True)
            + cells(FRESH2, method="hard_ste", lr=0.03))


def robust_q6f4():
    """Stopping rule (a): exact a2 beta 2 lr 1e-2 meets the win definition at 8/4 on fresh seeds 5-24 -> one more
    setting for robustness: the same cell, unchanged, at 6/4 vs hard STE's best 6/4 lr (0.02) on fresh seeds 5-14."""
    return (cells(FRESH, method="exact", lr=1e-2, alpha_end=2, beta=2, bits=[6, 4], diagnostics=True)
            + cells(FRESH, method="hard_ste", lr=0.02, bits=[6, 4]))


fn_map = {"phase3": phase3}

PHASES = {"phase1": phase1, "phase2_hard": phase2_hard,
          "phase2_stair_q6f4": lambda: phase2_stair([6, 4]), "phase2_stair_q6f3": lambda: phase2_stair([6, 3]),
          "phase2_stair_q4f2": lambda: phase2_stair([4, 2]), "ext_q4f2_exact": ext_q4f2_exact,
          "warmup": warmup, "lead_q8f4": lead_q8f4,
          "final_q6f3_curriculum": final_q6f3_curriculum,
          "extra_diag_beta": extra_diag_beta,
          "exact_beta_fixed": exact_beta_fixed,
          "exact_beta_ramp": exact_beta_ramp,
          "hybrid_b1": hybrid_b1,
          "exact_beta_ext": exact_beta_ext,
          "hybrid_beta": hybrid_beta,
          "exact_lead": exact_lead, "fresh_exact_lead": fresh_exact_lead,
          "fresh_exact_lead2": fresh_exact_lead2,
          "fresh2_q8f4": fresh2_q8f4,
          "robust_q6f4": robust_q6f4}

if __name__ == "__main__":
    os.makedirs(TASK_DIR, exist_ok=True)
    tasks = [t for p in sys.argv[1].split("+") for t in (PHASES[p] if p in PHASES else from_file(p))()]
    path = os.path.join(TASK_DIR, sys.argv[1] + ".json")
    json.dump(tasks, open(path, "w"), indent=1)
    print(f"{len(tasks)} tasks -> {path}")
