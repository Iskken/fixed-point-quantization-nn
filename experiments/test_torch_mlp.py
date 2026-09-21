import numpy as np
import torch

from src.models.mlp import MLP
from src.models.torch_mlp import TorchMLP, train
from src.quantization.quantize import fixed_point_quantize as np_quantize
from src.quantization.torch_quantize import fixed_point_quantize as torch_quantize

"""
Parity checks for the PyTorch port: the torch model must reproduce the
NumPy model's numbers, otherwise every result from here on is not
comparable to the earlier experiments.

python -m experiments.test_torch_mlp
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
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


print("=== 1. quantizer parity ===")
vals = rng.uniform(-12, 12, size=5000)
np_q = np_quantize(vals, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
torch_q = torch_quantize(torch.as_tensor(vals, dtype=torch.float64),
                         total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS).numpy()
check("torch quantizer == numpy quantizer", np.array_equal(np_q, torch_q),
      f"max|diff|={np.abs(np_q - torch_q).max():.3e}")

# round-half-to-even behaviour on exact .5 ties, where implementations often differ
ties = (np.arange(-20, 21) + 0.5) / (2 ** FRAC_BITS)
check("tie-breaking matches on exact .5 cases",
      np.array_equal(np_quantize(ties, TOTAL_BITS, FRAC_BITS),
                     torch_quantize(torch.as_tensor(ties), TOTAL_BITS, FRAC_BITS).numpy()))

# the straight-through flag: opt-in gradient through the rounding step
xg = torch.tensor([0.31, 0.62, -0.44], requires_grad=True)
ste = torch_quantize(xg, TOTAL_BITS, FRAC_BITS, straight_through=True)
ste.sum().backward()
check("straight_through=True gives STE gradients", torch.equal(xg.grad, torch.ones(3)),
      f"grad={xg.grad.tolist()}")
check("straight_through=False blocks the gradient",
      not torch_quantize(torch.tensor([0.31], requires_grad=True), TOTAL_BITS, FRAC_BITS).requires_grad)
check("straight_through flag does not change the values",
      torch.equal(ste.detach(), torch_quantize(xg.detach(), TOTAL_BITS, FRAC_BITS)))

print("\n=== 2. model parity against the trained NumPy checkpoint ===")
np_model = MLP.load(CHECKPOINT_PATH)
t_model = TorchMLP.from_numpy_checkpoint(CHECKPOINT_PATH)

check("architecture matches", np_model.layer_sizes == t_model.layer_sizes,
      f"{t_model.layer_sizes}")

X = rng.uniform(-1, 1, size=(300, np_model.layer_sizes[0]))
Xt = torch.as_tensor(X, dtype=torch.float64)

np_float = np_model.predict(X)
t_float = t_model.predict(Xt).numpy()
check("float predictions match", np.allclose(np_float, t_float, atol=1e-12),
      f"max|diff|={np.abs(np_float - t_float).max():.3e}")

np_quant = np_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
with torch.no_grad():
    t_quant = t_model.forward_quantized(Xt, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS).squeeze().numpy()
check("quantized predictions match", np.allclose(np_quant, t_quant, atol=1e-12),
      f"max|diff|={np.abs(np_quant - t_quant).max():.3e}")

print("\n=== 3. prefix consistency (the primitive the new method relies on) ===")
with torch.no_grad():
    full_prefix = t_model.forward_quantized_prefix(Xt, t_model.n_layers - 1,
                                                   total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS).squeeze().numpy()
check("prefix through last layer == forward_quantized",
      np.allclose(full_prefix, t_quant, atol=1e-15),
      f"max|diff|={np.abs(full_prefix - t_quant).max():.3e}")

partial_full = t_model.predict_partially_quantized(Xt, t_model.n_layers - 1,
                                                   total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS).numpy()
check("predict_partially_quantized(last) == quantized prediction",
      np.allclose(partial_full, t_quant, atol=1e-15))

partial_none = t_model.predict_partially_quantized(Xt, -1,
                                                   total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS).numpy()
check("upto=-1 quantizes the input only (differs from pure float)",
      not np.allclose(partial_none, t_float))

# every quantized activation must sit exactly on the fixed-point grid
with torch.no_grad():
    mid = t_model.forward_quantized_prefix(Xt, 1, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
check("prefix output lies on the fixed-point grid",
      torch.allclose(mid * (2 ** FRAC_BITS), torch.round(mid * (2 ** FRAC_BITS))),
      f"distinct levels={len(torch.unique(mid))}")

print("\n=== 4. training parity: torch full-batch SGD vs numpy gradient descent ===")
y = rng.uniform(-1, 1, size=300)
yt = torch.as_tensor(y, dtype=torch.float64)

np_small = MLP([4, 8, 1])
t_small = TorchMLP([4, 8, 1])
with torch.no_grad():
    for i in range(np_small.n_layers):
        t_small.layers[i].weight.copy_(torch.as_tensor(np_small.weights[i].T))
        t_small.layers[i].bias.copy_(torch.as_tensor(np_small.biases[i]))

np_small.fit(X, y, epochs=200, lr=0.05, verbose=False)
train(t_small, t_small.layer_parameters(0), Xt, yt, epochs=200, lr=0.05, optimizer_name="sgd")

np_after = np_small.predict(X)
t_after = t_small.predict(Xt).numpy()
check("200 epochs of SGD match numpy gradient descent",
      np.allclose(np_after, t_after, atol=1e-9),
      f"max|diff|={np.abs(np_after - t_after).max():.3e}")

print(f"\n{passed} passed, {failed} failed")
