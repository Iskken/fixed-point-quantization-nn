import numpy as np
import torch

from src.models.mlp import MLP
from src.models.torch_qat import QATMLP, train_qat, alpha_beta_at, learning_rate_at
from src.quantization.torch_stair import steep_fraction

"""
Parity gate for the PyTorch QAT model: with grad_mode="clamped",
mask_saturation=False and quantize_activations=False it must reproduce the
NumPy MLP.forward_qat_non_linear / backward_qat_non_linear / fit_qat_non_linear
numbers. Only after this passes are differences between torch and NumPy
experiments attributable to the experiments and not to the port.

python -m experiments.test_torch_qat
"""

TOTAL_BITS, FRAC_BITS = 8, 4
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


def paired_models(sizes, activation="tanh"):
    np.random.seed(1)
    np_model = MLP(sizes)
    t_model = QATMLP(sizes, activation=activation)
    with torch.no_grad():
        for i in range(np_model.n_layers):
            t_model.layers[i].weight.copy_(torch.as_tensor(np_model.weights[i].T))
            t_model.layers[i].bias.copy_(torch.as_tensor(np_model.biases[i]))
    return np_model, t_model


X = rng.uniform(-1, 1, size=(200, 4))
y = np.sin(3 * X[:, 0]) + 0.3 * X[:, 2] * X[:, 3]
Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
PARITY = dict(grad_mode="clamped", mask_saturation=False, quantize_activations=False)

print("=== 1. single forward / backward parity ===")
for alpha in (1.0, 5.0, 10.0, 100.0):
    np_model, t_model = paired_models([4, 16, 16, 1])
    y_np = np_model.forward_qat_non_linear(X, TOTAL_BITS, FRAC_BITS, alpha, 1.0)
    np_model.backward_qat_non_linear(X, y, y_np)

    out = t_model.forward_qat(Xt, TOTAL_BITS, FRAC_BITS, alpha, 1.0, **PARITY)
    torch.nn.functional.mse_loss(out, yt.reshape(-1, 1)).backward()

    f_diff = np.abs(y_np - out.detach().numpy()).max()
    g_diff = max(
        max(np.abs(np_model.grad_weights[i] - t_model.layers[i].weight.grad.numpy().T).max(),
            np.abs(np_model.grad_biases[i] - t_model.layers[i].bias.grad.numpy()).max())
        for i in range(np_model.n_layers))
    check(f"alpha={alpha}: forward and all gradients match",
          f_diff < 1e-12 and g_diff < 1e-12, f"forward {f_diff:.1e}, grad {g_diff:.1e}")

print("\n=== 2. training parity (300 epochs, alpha 1 -> 10) ===")
np_model, t_model = paired_models([4, 16, 16, 1])
EPOCHS = 300
np_loss, *_ = np_model.fit_qat_non_linear(
    X, y, epochs=EPOCHS, lr=0.05, total_bits=TOTAL_BITS, frac_bits=FRAC_BITS,
    schedule_type="alpha_linear", alpha_start=1.0, alpha_end=10.0,
    transition_epochs=EPOCHS, verbose=False)
h = train_qat(t_model, Xt, yt, Xt, yt, max_epochs=EPOCHS, lr=0.05,
              schedule_type="alpha_linear", alpha_start=1.0, alpha_end=10.0,
              transition_epochs=EPOCHS, patience=10 ** 9, **PARITY)
check("loss history matches NumPy", np.allclose(np_loss, h["train_loss"], atol=1e-10),
      f"max|diff|={np.abs(np.array(np_loss) - np.array(h['train_loss'])).max():.1e}")
w_diff = max(np.abs(np_model.weights[i] - t_model.layers[i].weight.detach().numpy().T).max()
             for i in range(np_model.n_layers))
check("final weights match NumPy", w_diff < 1e-9, f"max|diff|={w_diff:.1e}")

print("\n=== 3. schedules ===")
check("alpha_linear endpoints", alpha_beta_at(0.0, "alpha_linear", 1, 10) == (1.0, 1.0)
      and alpha_beta_at(1.0, "alpha_linear", 1, 10) == (10.0, 1.0))
np_m = MLP([1, 1])
ok = all(np.allclose(alpha_beta_at(p, s, 1.0, 10.0), np_m._get_scheduled_alpha_beta(p, s, 1.0, 10.0))
         for p in np.linspace(0, 1, 21) for s in ("alpha_linear", "alpha_step", "alpha_warm_up"))
check("alpha schedules match NumPy", ok)
check("alpha_dependent lr = lr0*(a0/a)^0.5 like NumPy",
      np.isclose(learning_rate_at(0, 10, "alpha_dependent", 0.01, 4.0, 1.0, 1.0),
                 np_m._get_learning_rate(0, 10, "alpha_dependent", 0.01, 4.0, 1.0, 1.0)))

print("\n=== 4. new behaviour: validation, early stopping, diagnostics ===")
_, t_model = paired_models([4, 16, 1])
h = train_qat(t_model, Xt, yt, Xt, yt, max_epochs=3000, lr=0.05, transition_epochs=200,
              alpha_start=1.0, alpha_end=10.0, patience=50, min_delta=1e-4,
              snapshot_alphas=[5.0, 10.0])
check("early stopping triggers after the transition", h["converged"] and h["stopped_epoch"] >= 200,
      f"stopped at {h['stopped_epoch']}, best epoch {h['best_epoch']}")
check("best exact val MSE is restored", np.isclose(
    torch.nn.functional.mse_loss(t_model.forward_exact(Xt), yt.reshape(-1, 1)).item(),
    h["best_val_exact_mse"]))
check("per-layer grad stats recorded", len(h["grad_stats"][0]) == 2 and "norm" in h["grad_stats"][0][0])
check("steep fractions in [0, 1]", all(0 <= f <= 1 for row in h["steep_fraction"] for f in row))
check("gradient snapshots captured", set(h["grad_snapshots"]) == {5.0, 10.0})

w = torch.tensor([0.0, 0.0625, 0.4985 * 0.0625, 0.02], dtype=torch.float64)
check("steep_fraction counts only far-from-grid entries", steep_fraction(w, 4, 10.0, 1.0) == 0.25,
      f"{steep_fraction(w, 4, 10.0, 1.0)}")

print("\n=== 5. ReLU and activation quantization ===")
_, relu_model = paired_models([4, 16, 1], activation="relu")
out = relu_model.forward_qat(Xt, TOTAL_BITS, FRAC_BITS, 5.0, 1.0, grad_mode="exact", quantize_activations=True)
out.sum().backward()
check("relu + quantized activations: finite output and gradients",
      torch.isfinite(out).all() and all(torch.isfinite(l.weight.grad).all() for l in relu_model.layers))

_, a_model = paired_models([4, 16, 1])
ref = a_model.forward_exact(Xt)
soft = a_model.forward_qat(Xt, TOTAL_BITS, FRAC_BITS, 1e5, 1.0, grad_mode="exact", quantize_activations=True)
check("large alpha with activation quantization approaches the exact forward",
      (soft - ref).abs().max().item() < 0.5 * 2 ** -FRAC_BITS * 4,
      f"max|diff|={(soft - ref).abs().max().item():.4f}")

print(f"\n{passed} passed, {failed} failed")
