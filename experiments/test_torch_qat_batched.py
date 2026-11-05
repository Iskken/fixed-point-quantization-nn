import numpy as np
import torch

from experiments.torch_qat.common import make_data, make_model
from src.models.torch_qat import train_qat
from src.models.torch_qat_batched import train_qat_batched

"""
Parity gate for the batched trainer: on CPU in float64, every run of a train_qat_batched batch must
reproduce train_qat run on its own (curves, best/stopped epoch, flags, final weights). Only after
this passes can a difference between GPU and CPU runs be put down to the hardware, not the port.

The batched weight-gradient matmul sums over the samples in a different order than the 2D one, so
weight gradients differ by ~1e-17 (forward, loss and bias gradients are bitwise equal). Exact- and
slope-1 runs amplify that noise (it grows smoothly from 1e-16 to 1e-3 over ~100 epochs, for a batch
of one run as well as of four), so they are compared over the first epochs only (section 1). The
logic (mixed lrs / alphas / lr schedule, runs stopping at different epochs and leaving the batch,
saved states, abort) is checked strictly on runs whose trajectories stay stable (sections 2-5).

python -m experiments.test_torch_qat_batched
"""

TOL = 1e-10
passed, failed = 0, 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def max_diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    return float(np.nanmax(np.where(both_nan, 0.0, np.abs(a - b)))) if a.size else 0.0


def compare(label, single, batched, tol=TOL):
    h1, m1 = single
    h2, s2 = batched
    curves = ["train_loss", "val_qat_loss", "val_exact_mse", "test_exact_mse", "lr", "alpha"]
    d = max(max_diff(h1[k], h2[k]) for k in curves)
    g1, g2 = ([[[l[s] for s in ("mean", "median", "max", "min", "norm")] for l in e] for e in h["grad_stats"]]
              for h in (h1, h2))
    d_grad = max_diff(g1, g2) / max(np.abs(g1).max(), 1.0)   # relative: exact-slope gradients reach ~1e3
    d_steep = max_diff(h1["steep_fraction"], h2["steep_fraction"])
    d_w = max((m1.state_dict()[k] - s2[k]).abs().max().item() for k in s2)
    same = all(h1[k] == h2[k] for k in ("best_epoch", "stopped_epoch", "converged", "aborted"))
    check(label, same and max(d, d_grad, d_steep, d_w) < tol,
          f"epochs {h1['stopped_epoch']}/{h2['stopped_epoch']} best {h1['best_epoch']}/{h2['best_epoch']} "
          f"curves {d:.1e} grads {d_grad:.1e} steep {d_steep:.1e} weights {d_w:.1e}")


def run_both(runs, **kw):
    single = []
    for r in runs:
        m = make_model(0, activation="relu")
        m.load_state_dict({k: v.clone() for k, v in r["state"].items()})
        d = r["data"]
        h = train_qat(m, d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_test=d["X_test"], y_test=d["y_test"],
                      lr=r["lr"], alpha_end=r.get("alpha_end", 10.0), grad_mode=r.get("grad_mode", "exact"),
                      learning_rate_schedule=r.get("lr_schedule"), **kw)
        single.append((h, m))
    return single, train_qat_batched(runs, **kw)


def run(seed, lr, **extra):
    return {"state": {k: v.clone() for k, v in make_model(seed, activation="relu").state_dict().items()},
            "data": make_data(seed), "lr": lr, **extra}


torch.set_num_threads(1)
COMMON = dict(max_epochs=400, transition_epochs=100, patience=40, min_delta=1e-3, quantize_activations=True,
              mask_saturation=True, save_state_at=[50, 150])

print("=== 1. stair, Adam: exact / slope 1 / clamped in one batch, first 15 epochs ===")
runs = [run(0, 1e-3, grad_mode="exact", alpha_end=10.0),
        run(1, 1e-2, grad_mode="ste", alpha_end=30.0),
        run(2, 1e-3, grad_mode="clamped", alpha_end=5.0),
        run(0, 3e-3, grad_mode="exact", alpha_end=30.0, lr_schedule="alpha_dependent")]
single, batched = run_both(runs, optimizer_name="adam", lr_power=1.0, **dict(COMMON, max_epochs=15, transition_epochs=10))
for i, (s, b) in enumerate(zip(single, batched)):
    compare(f"stair run {i} ({runs[i]['grad_mode']})", s, b)

print("=== 2. clamped stair, Adam: mixed lrs / alphas / lr schedule, runs stop at different epochs ===")
runs = [run(0, 1e-3, grad_mode="clamped", alpha_end=10.0),
        run(1, 3e-3, grad_mode="clamped", alpha_end=30.0),
        run(2, 1e-3, grad_mode="clamped", alpha_end=5.0),
        run(0, 3e-3, grad_mode="clamped", alpha_end=30.0, lr_schedule="alpha_dependent")]
single, batched = run_both(runs, optimizer_name="adam", lr_power=1.0, **COMMON)
for i, (s, b) in enumerate(zip(single, batched)):
    compare(f"clamped run {i}", s, b)
check("runs stop at different epochs", len({h["stopped_epoch"] for h, _ in batched}) > 1,
      str([h["stopped_epoch"] for h, _ in batched]))
check("saved states match", all(max((s[0]["states"][e][k] - b[0]["states"][e][k]).abs().max().item()
                                    for e in s[0]["states"] for k in b[0]["states"][e]) < TOL
                                for s, b in zip(single, batched)))

print("=== 3. hard STE, Adam ===")
single, batched = run_both([run(0, 3e-3), run(1, 1e-2), run(2, 3e-3)], hard_ste=True, optimizer_name="adam", **COMMON)
for i, (s, b) in enumerate(zip(single, batched)):
    compare(f"hard STE run {i}", s, b)

print("=== 4. stair, SGD: all grad modes over the full run (SGD keeps the ~1e-17 gradient noise below the last bit) ===")
single, batched = run_both([run(0, 1e-2, grad_mode="clamped"), run(1, 3e-2, grad_mode="exact", alpha_end=3.0),
                            run(1, 1e-2, grad_mode="ste", alpha_end=30.0), run(2, 1e-2, grad_mode="exact", alpha_end=10.0)],
                           optimizer_name="sgd", **COMMON)
for i, (s, b) in enumerate(zip(single, batched)):
    # run 3: exact slope at alpha 10 multiplies the gradient by up to 10 per stair (7 stairs deep), which
    # lifts the ~1e-17 noise to ~1e-8 (curves) and ~1e-6 (gradient stats); same epochs and flags required
    compare(f"SGD run {i}", s, b, tol=1e-5 if i == 3 else TOL)

print("=== 5. abort rule ===")
single, batched = run_both([run(0, 1e-3, grad_mode="exact"), run(1, 1e-3, grad_mode="ste")], optimizer_name="adam",
                           abort_val_above=0.0, abort_epoch=10, **COMMON)
for i, (s, b) in enumerate(zip(single, batched)):
    compare(f"aborted run {i}", s, b)

print("=== 6. per-run ramp length and patience (SGD, strict) ===")
runs = [run(0, 1e-2, grad_mode="exact", alpha_end=10.0, transition=50, patience=30),
        run(1, 1e-2, grad_mode="ste", alpha_end=30.0, transition=150, patience=60)]
single = []
for r in runs:
    m = make_model(0, activation="relu")
    m.load_state_dict({k: v.clone() for k, v in r["state"].items()})
    d = r["data"]
    h = train_qat(m, d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_test=d["X_test"], y_test=d["y_test"],
                  lr=r["lr"], alpha_end=r["alpha_end"], grad_mode=r["grad_mode"], optimizer_name="sgd",
                  **dict(COMMON, transition_epochs=r["transition"], patience=r["patience"]))
    single.append((h, m))
batched = train_qat_batched(runs, optimizer_name="sgd", **COMMON)
for i, (s_, b) in enumerate(zip(single, batched)):
    compare(f"ramp {runs[i]['transition']} / patience {runs[i]['patience']}", s_, b)

print("=== 7. stair beta != 1 and flat-zone gradient floor vs stair_quantize ===")
from src.quantization.torch_stair import stair_quantize, stair_geometry
from src.models.torch_qat_batched import _BatchedStair, _geometry
x = torch.linspace(-9, 9, 20001, dtype=torch.float64)
for alpha, beta, floor in [(10.0, 2.0, None), (30.0, 0.5, None), (10.0, 1.0, 0.3), (100.0, 1.0, 0.1)]:
    geo = [torch.tensor(g, dtype=torch.float64).reshape(1, 1, 1) for g in _geometry(1 / 16, alpha, beta, "exact", floor)]
    xb = x.clone().reshape(1, 1, -1).requires_grad_()
    out_b = _BatchedStair.apply(xb, *geo, 8, 4, True)
    out_b.sum().backward()
    xs = x.clone().requires_grad_()
    out_s = stair_quantize(xs, 8, 4, alpha, beta, grad_mode="exact")
    out_s.sum().backward()
    g_ref = xs.grad.clone()
    if floor is not None:   # floor only lifts the flat-zone slope, where the exact slope is c/a
        flat = stair_geometry(1 / 16, alpha, beta)[4]
        g_ref = torch.where(g_ref == flat, torch.full_like(g_ref, max(flat, floor)), g_ref)
    check(f"alpha {alpha:g} beta {beta:g} floor {floor}",
          (out_b.flatten() - out_s).abs().max().item() == 0 and (xb.grad.flatten() - g_ref).abs().max().item() == 0)

print("=== 8. float continuation vs continue_float (SGD) ===")
from experiments.torch_qat.common import continue_float
r = run(2, 1e-2)
m = make_model(2, activation="relu")
m.load_state_dict({k: v.clone() for k, v in r["state"].items()})
h1 = continue_float(m, r["data"], max_epochs=400, lr=1e-2, patience=40, min_delta=1e-3)
(h2, s2), = train_qat_batched([r], max_epochs=400, patience=40, min_delta=1e-3, float_train=True,
                              quantize_activations=True, optimizer_name="sgd")
d = max(max_diff(h1[k], h2[k]) for k in ("train_loss", "val_qat_loss", "val_exact_mse", "test_exact_mse"))
d_w = max((m.state_dict()[k] - s2[k]).abs().max().item() for k in s2)
check("float continuation", h1["best_epoch"] == h2["best_epoch"] and h1["stopped_epoch"] == h2["stopped_epoch"]
      and max(d, d_w) < TOL, f"epochs {h1['stopped_epoch']}/{h2['stopped_epoch']} curves {d:.1e} weights {d_w:.1e}")

print("=== 9. diagnostics do not change training ===")
runs = [run(0, 1e-3, grad_mode="exact"), run(1, 1e-2, grad_mode="ste", alpha_end=30.0)]
kw = dict(COMMON, max_epochs=60, transition_epochs=30)
plain = train_qat_batched(runs, optimizer_name="adam", **kw)
diag = train_qat_batched(runs, optimizer_name="adam", diagnostics=True, **kw)
same = all(a[0]["val_exact_mse"] == b[0]["val_exact_mse"] for a, b in zip(plain, diag))
vals = [x for hh, _ in diag for k in ("sat_params", "sat_preact", "sat_act", "wq_change") for x in hh[k]
        if x == x]
check("curves identical with diagnostics on", same)
check("diagnostic values are shares in [0, 1]", len(vals) > 0 and min(vals) >= 0 and max(vals) <= 1,
      f"wq_change epoch 1-5 run 0: {[round(x, 3) for x in diag[0][0]['wq_change'][1:6]]}")

print("=== 10a. lr warm-up multiplies the scheduled lr; warmup 0 changes nothing ===")
runs = [run(0, 1e-2, grad_mode="ste", alpha_end=10.0, warmup=20), run(1, 1e-3, grad_mode="exact", alpha_end=30.0,
        lr_schedule="alpha_dependent", warmup=10), run(2, 1e-3, grad_mode="exact", warmup=0)]
kw = dict(COMMON, max_epochs=40, transition_epochs=30)
out = train_qat_batched(runs, optimizer_name="adam", lr_power=1.0, **kw)
ref = train_qat_batched([dict(r, warmup=0) for r in runs], optimizer_name="adam", lr_power=1.0, **kw)
ratio_ok = all(abs(out[i][0]["lr"][e] / ref[i][0]["lr"][e] - min(1.0, (e + 1) / w)) < 1e-12 if w else
               out[i][0]["lr"][e] == ref[i][0]["lr"][e] for i, w in enumerate((20, 10, 0)) for e in range(40))
check("lr = scheduled lr x min(1, (epoch+1)/warmup)", ratio_ok)
check("warmup 0 run unchanged", out[2][0]["val_exact_mse"] == ref[2][0]["val_exact_mse"])

print("=== 10b. beta ramp: beta 1 -> beta with the alpha ramp; identity at epoch 0; beta 1 unchanged ===")
runs = [run(0, 1e-3, grad_mode="exact", alpha_end=3.0, beta=5.0, beta_ramp=True),
        run(1, 1e-3, grad_mode="exact", alpha_end=3.0, beta=1.0, beta_ramp=True), run(1, 1e-3, grad_mode="exact", alpha_end=3.0)]
kw = dict(COMMON, max_epochs=40, transition_epochs=30)
out = train_qat_batched(runs, optimizer_name="adam", **kw)
hb = out[0][0]["beta"]
check("beta follows 1 + (beta - 1) * progress", all(abs(hb[e] - (1 + 4 * min(e / 29, 1))) < 1e-12 for e in range(40)),
      f"beta at 0 / 15 / 39: {hb[0]:g} / {hb[15]:.4f} / {hb[39]:g}")
check("beta_ramp with beta 1 is identical to no ramp", out[1][0]["val_exact_mse"] == out[2][0]["val_exact_mse"])

print("=== 10c. hybrid (hard forward, stair backward) ===")
from src.models.torch_qat_batched import _BatchedStair
from src.quantization.torch_quantize import fixed_point_quantize
xs = (torch.randn(2, 64, 64, dtype=torch.float64) * 4).requires_grad_()
alpha, beta = 3.0, 3.0
geo = [torch.tensor(g, dtype=torch.float64).reshape(1, 1, 1).expand(2, 1, 1).contiguous()
       for g in _geometry(1 / 16, alpha, beta, "exact")]
hard = torch.tensor([True, False]).reshape(2, 1, 1)
out_h = _BatchedStair.apply(xs, *geo, 8, 4, True, hard)
out_h.sum().backward()
xr = xs.detach()[0].clone().requires_grad_()
fixed_point_quantize(xr, 8, 4, straight_through=True).sum().backward()
inside = xr.grad != 0   # hard STE's mask: the rounded value lies in the range (a bit wider than the stair's)
g = stair_geometry(1 / 16, alpha, beta)
x0 = xs.detach()[0]
is_flat = (x0 - torch.round(x0 * 16) / 16).abs() <= g[0] / 2
expected = torch.where(is_flat, torch.tensor(g[4], dtype=torch.float64), torch.tensor(g[5], dtype=torch.float64)) * inside
check("hybrid forward = fixed-point rounding", torch.equal(out_h[0].detach(), fixed_point_quantize(x0, 8, 4)))
check("hybrid backward = stair exact slope where hard STE passes the gradient, 0 where the rounding saturates",
      torch.equal(xs.grad[0], expected), f"{int((~inside).sum())} saturated entries")
xs2 = xs.detach()[1].clone().requires_grad_()
ref = stair_quantize(xs2, 8, 4, alpha, beta, grad_mode="exact")
ref.sum().backward()
check("non-hybrid run in the same call unchanged", torch.equal(out_h[1].detach(), ref.detach()) and torch.equal(xs.grad[1], xs2.grad))
runs = [run(0, 1e-2, grad_mode="ste", hard_forward=True, transition=1), run(1, 3e-2, grad_mode="ste", hard_forward=True, transition=1)]
kw = dict(COMMON, max_epochs=300, patience=10 ** 6)
hyb = train_qat_batched(runs, optimizer_name="adam", **kw)
hst = train_qat_batched([dict(r, hard_forward=False) for r in runs], optimizer_name="adam", hard_ste=True, **kw)
check("hybrid with slope-1 backward reproduces hard STE (300 epochs, Adam)",
      all(hyb[i][0]["val_exact_mse"] == hst[i][0]["val_exact_mse"] for i in range(2)),
      f"max diff {max(np.abs(np.array(hyb[i][0]['val_exact_mse']) - np.array(hst[i][0]['val_exact_mse'])).max() for i in range(2)):.1e}")

if torch.cuda.is_available():
    print("=== 10. GPU float64 vs CPU (first 300 epochs, no stopping) ===")
    runs = [run(0, 1e-3, grad_mode="exact"), run(1, 1e-2, grad_mode="ste")]
    kw = dict(COMMON, max_epochs=300, patience=10 ** 6)
    cpu = train_qat_batched(runs, optimizer_name="adam", **kw)
    gpu = train_qat_batched(runs, optimizer_name="adam", device="cuda", **kw)
    for i, ((hc, _), (hg, _)) in enumerate(zip(cpu, gpu)):
        d = np.abs(np.array(hc["val_exact_mse"]) - np.array(hg["val_exact_mse"]))
        print(f"  info  run {i}: max |val_exact GPU - CPU| = {d.max():.1e}, first epoch above 1e-9: "
              f"{int(np.argmax(d > 1e-9)) if (d > 1e-9).any() else None}")

print(f"\n{passed} passed, {failed} failed")
