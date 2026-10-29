import numpy as np
import torch

from src.models.mlp import MLP
from src.quantization.quantize import fixed_point_quantize as np_quantize
from src.quantization.torch_stair import stair_quantize

"""
Checks for the PyTorch stair quantizer (src/quantization/torch_stair.py):
parity with the NumPy MLP._stair_quantize, limiting cases, gradcheck of the
exact mode, and the documented behaviour of the clamped / STE modes.

python -m experiments.test_torch_stair
"""

TOTAL_BITS, FRAC_BITS = 8, 4
LSB = 2.0 ** (-FRAC_BITS)
ALPHA_BETA = [(0.1, 1.0), (1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (100.0, 1.0),
              (3.0, 0.2), (10.0, 5.0)]

rng = np.random.default_rng(0)
passed, failed = 0, 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


np_mlp = MLP([1, 1])
x_np = rng.uniform(-9, 9, size=20000)       # includes saturation (range is +-8)
x_t = torch.as_tensor(x_np, dtype=torch.float64)

print("=== 1. forward parity with NumPy _stair_quantize ===")
for alpha, beta in ALPHA_BETA:
    ref, ref_grad = np_mlp._stair_quantize(x_np, TOTAL_BITS, FRAC_BITS, alpha, beta)
    out = stair_quantize(x_t, TOTAL_BITS, FRAC_BITS, alpha, beta).numpy()
    check(f"forward alpha={alpha} beta={beta}", np.allclose(ref, out, atol=1e-12),
          f"max|diff|={np.abs(ref - out).max():.2e}")

print("\n=== 2. clamped mode reproduces the NumPy surrogate gradient ===")
for alpha, beta in ALPHA_BETA:
    _, ref_grad = np_mlp._stair_quantize(x_np, TOTAL_BITS, FRAC_BITS, alpha, beta)
    xg = x_t.clone().requires_grad_(True)
    stair_quantize(xg, TOTAL_BITS, FRAC_BITS, alpha, beta,
                   grad_mode="clamped", mask_saturation=False).sum().backward()
    check(f"clamped grad alpha={alpha} beta={beta}",
          np.allclose(ref_grad, xg.grad.numpy(), atol=1e-12),
          f"max|diff|={np.abs(ref_grad - xg.grad.numpy()).max():.2e}")

print("\n=== 3. exact mode: gradcheck and finite differences ===")
# keep samples away from segment boundaries, where the derivative is undefined
inside = x_t[(x_t.abs() < 7)]
centers = torch.round(inside / LSB) * LSB
frac = (inside - centers).abs() / LSB
for alpha, beta in [(1.0, 1.0), (5.0, 1.0), (10.0, 1.0), (3.0, 0.2)]:
    a_w = alpha / (alpha + 1.0)             # flat-zone half-width is a_w/2 (in lsb)
    away = (frac - a_w / 2).abs() > 1e-3
    away &= (frac - 0.5).abs() > 1e-3
    pts = inside[away][:200].clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda t: stair_quantize(t, TOTAL_BITS, FRAC_BITS, alpha, beta, grad_mode="exact"),
        (pts,), eps=1e-7, atol=1e-5)
    check(f"gradcheck exact alpha={alpha} beta={beta}", ok)

print("\n=== 4. documented behaviour of the three modes ===")
alpha = 10.0
steep_pt = torch.tensor([0.4985 * LSB + 3 * LSB], dtype=torch.float64)   # steep zone
flat_pt = torch.tensor([0.01 * LSB + 3 * LSB], dtype=torch.float64)      # flat zone
for mode, (want_flat, want_steep) in {"exact": (1 / alpha, alpha),
                                      "clamped": (0.1, 1.0),
                                      "ste": (1.0, 1.0)}.items():
    got = []
    for pt in (flat_pt, steep_pt):
        p = pt.clone().requires_grad_(True)
        stair_quantize(p, TOTAL_BITS, FRAC_BITS, alpha, 1.0, grad_mode=mode).sum().backward()
        got.append(p.grad.item())
    check(f"{mode}: (flat, steep) slope at alpha=10",
          np.allclose(got, [want_flat, want_steep], rtol=1e-9),
          f"got={np.round(got, 4).tolist()} want={[want_flat, want_steep]}")

print("\n=== 5. limiting cases ===")
out_id = stair_quantize(x_t[np.abs(x_np) < 7], TOTAL_BITS, FRAC_BITS, 1.0, 1.0).numpy()
check("alpha=beta=1 is the identity inside the range",
      np.allclose(out_id, x_np[np.abs(x_np) < 7], atol=1e-12))

hard = np_quantize(x_np, TOTAL_BITS, FRAC_BITS)
errs = []
for alpha in (10.0, 100.0, 1e4, 1e6):
    soft = stair_quantize(x_t, TOTAL_BITS, FRAC_BITS, alpha, 1.0).numpy()
    errs.append(np.abs(soft - hard).max() / LSB)
check("max deviation from the hard quantizer stays <= 0.5 lsb",
      all(e <= 0.5 + 1e-9 for e in errs), f"deviation(lsb)={np.round(errs, 3).tolist()}")

grid = (np.arange(-120, 120)) * LSB
on_grid = stair_quantize(torch.as_tensor(grid), TOTAL_BITS, FRAC_BITS, 7.0, 0.3).numpy()
check("curve passes exactly through every grid point", np.allclose(on_grid, grid, atol=1e-12))

xs = torch.linspace(-2, 2, 200001, dtype=torch.float64)
ys = stair_quantize(xs, TOTAL_BITS, FRAC_BITS, 50.0, 1.0)
check("output is continuous (no jumps larger than the sampling step * max slope)",
      (ys[1:] - ys[:-1]).abs().max().item() <= 50.0 * (xs[1] - xs[0]).item() * 1.01)

print("\n=== 6. saturation masking ===")
big = torch.tensor([9.0, -9.0, 1.03], dtype=torch.float64)
for mask, want in ((True, [0.0, 0.0]), (False, None)):
    p = big.clone().requires_grad_(True)
    stair_quantize(p, TOTAL_BITS, FRAC_BITS, 5.0, 1.0, grad_mode="ste", mask_saturation=mask).sum().backward()
    if mask:
        check("gradient is zero outside the representable range (mask on)",
              p.grad[:2].tolist() == want and p.grad[2].item() == 1.0, f"grad={p.grad.tolist()}")
    else:
        check("gradient is not masked when mask_saturation=False",
              p.grad.tolist() == [1.0, 1.0, 1.0], f"grad={p.grad.tolist()}")

print("\n=== 7. float32 works too ===")
x32 = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
o32 = stair_quantize(x32, TOTAL_BITS, FRAC_BITS, 10.0, 1.0)
o32.sum().backward()
ref32 = np_mlp._stair_quantize(x_np, TOTAL_BITS, FRAC_BITS, 10.0, 1.0)[0]
check("float32 forward close to float64 reference", np.allclose(o32.detach().numpy(), ref32, atol=1e-5),
      f"max|diff|={np.abs(o32.detach().numpy() - ref32).max():.2e}, dtype={o32.dtype}")

print(f"\n{passed} passed, {failed} failed")
