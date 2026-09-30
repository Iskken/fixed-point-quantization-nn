import os
import tempfile

import torch
import torch.nn.functional as F

from src.models.torch_cnn import LayerwiseBits, TorchCNN, bits_to_cover, frac_bits
from src.quantization.torch_quantize import fixed_point_quantize

"""
Consistency checks for TorchCNN: the float, quantized and prefix paths must
agree exactly wherever they describe the same computation, otherwise the
layer-wise PTQ comparisons built on them mean nothing. Uses random weights
and inputs, so it needs no dataset download.

python -m experiments.test_torch_cnn
"""

TOTAL_BITS, FRAC_BITS = 8, 4

torch.manual_seed(0)
passed, failed = 0, 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def on_grid(t, f):
    scaled = t * 2.0 ** f
    return torch.equal(scaled, torch.round(scaled))


for name, channels, size, conv in [("MNIST-shaped", 1, 28, (16, 32)), ("SVHN-shaped", 3, 32, (32, 64))]:
    print(f"\n=== {name} model ===")
    model = TorchCNN(in_channels=channels, image_size=size, conv_channels=conv)
    x = torch.rand(64, channels, size, size)
    n = model.n_layers

    print("--- 1. float paths ---")
    logits = model(x)
    check("output shape is (batch, 10)", tuple(logits.shape) == (64, 10), str(tuple(logits.shape)))
    check("stage names", model.stage_names() == ["conv1", "conv2", "fc1", "fc2"], str(model.stage_names()))

    a = x
    split_ok = True
    for i in range(n):
        split_ok &= torch.equal(model.forward_from(a, i), logits)
        z = model.layers[i](model._enter(i, a))
        a = z if i == n - 1 else model._activate(i, z)
    check("forward_from(float output of stage i-1, i) == forward, for every i", split_ok)

    print("--- 2. quantized paths ---")
    fq = model.forward_quantized(x, TOTAL_BITS, FRAC_BITS)
    check("prefix through last stage == forward_quantized",
          torch.equal(model.forward_quantized_prefix(x, n - 1, TOTAL_BITS, FRAC_BITS), fq))
    check("predict_partially_quantized(last) == forward_quantized",
          torch.equal(model.predict_partially_quantized(x, n - 1, TOTAL_BITS, FRAC_BITS), fq))
    xq = fixed_point_quantize(x, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
    check("predict_partially_quantized(-1) == float net on the quantized input",
          torch.allclose(model.predict_partially_quantized(x, -1, TOTAL_BITS, FRAC_BITS), model(xq)))
    check("quantized logits lie on the fixed-point grid", on_grid(fq, FRAC_BITS))

    prefix_on_grid = all(on_grid(model.forward_quantized_prefix(x, i, TOTAL_BITS, FRAC_BITS), FRAC_BITS)
                         for i in range(-1, n))
    check("every prefix output lies on the grid", prefix_on_grid)

    # post-activation rounding is a no-op for ReLU + max-pool
    noop = True
    a = xq
    for i in range(n - 1):
        layer = model.layers[i]
        q = lambda t: fixed_point_quantize(t, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
        z = q(model._stage_op(i, model._enter(i, a), q(layer.weight), q(layer.bias)))
        act = model._activate(i, z)
        noop &= torch.equal(act, q(act))
        a = q(act)
    check("post-ReLU/pool rounding is a no-op (kept only for parity with the MLP)", noop)

    print("--- 3. per-layer formats ---")
    uniform = LayerwiseBits(FRAC_BITS, [FRAC_BITS] * n, [FRAC_BITS] * n)
    check("uniform LayerwiseBits == plain int fractional_bits",
          torch.equal(model.forward_quantized(x, TOTAL_BITS, uniform), fq))
    mixed = LayerwiseBits(7, [8, 7, 9, 6], [3, 2, 4, 3])
    check("frac_bits resolves per role",
          (frac_bits(mixed, "input"), frac_bits(mixed, "weights", 2), frac_bits(mixed, "activations", 1)) == (7, 9, 2))
    check("mixed-format prefix lands on each stage's own activation grid",
          all(on_grid(model.forward_quantized_prefix(x, i, TOTAL_BITS, mixed), mixed.activations[i]) for i in range(n)))

    alloc = model.allocate_bits(x, TOTAL_BITS)
    r = model.tensor_ranges(x)
    covers = all(m <= 2.0 ** (TOTAL_BITS - 1 - f)
                 for m, f in [(r["input"], alloc.input)] + list(zip(r["weights"], alloc.weights))
                 + list(zip(r["activations"], alloc.activations)))
    tight = all(m > 2.0 ** (TOTAL_BITS - 2 - f)
                for m, f in list(zip(r["weights"], alloc.weights)) + list(zip(r["activations"], alloc.activations)))
    check("allocate_bits covers every range (no clipping)", covers, str(alloc))
    check("... and is tight (one fewer integer bit would clip)", tight)

    alloc_mse = model.allocate_bits(x, 4, method="mse")
    tensors = model.calibration_tensors(x)
    alloc_max4 = model.allocate_bits(x, 4, method="max")

    def tensor_mse(t, f):
        return torch.mean((fixed_point_quantize(t, total_bits=4, fractional_bits=f) - t) ** 2).item()

    pairs = [(tensors["input"], alloc_mse.input, alloc_max4.input)] + \
        list(zip(tensors["weights"], alloc_mse.weights, alloc_max4.weights)) + \
        list(zip(tensors["activations"], alloc_mse.activations, alloc_max4.activations))
    check("mse allocation never has more quantization error than max allocation (4-bit)",
          all(tensor_mse(t, fm) <= tensor_mse(t, fx) for t, fm, fx in pairs), str(alloc_mse))
    check("... and only ever adds fractional bits (clips, never widens the range)",
          all(fm >= fx for _, fm, fx in pairs))

    print("--- 4. layer-wise helpers ---")
    before = model.forward_quantized(x, TOTAL_BITS, FRAC_BITS)
    m2 = TorchCNN(in_channels=channels, image_size=size, conv_channels=conv)
    m2.load_state_dict(model.state_dict())
    m2.quantize_layer_(1, TOTAL_BITS, FRAC_BITS)
    check("quantize_layer_ puts that stage's weights on the grid",
          all(on_grid(p, FRAC_BITS) for p in m2.layers[1].parameters()))
    check("quantizing weights in place leaves the quantized forward unchanged",
          torch.equal(m2.forward_quantized(x, TOTAL_BITS, FRAC_BITS), before))

    # the quantized-activation training step: gradient reaches stages >= i only
    i = 2
    for j in range(i):
        m2.freeze_layer_(j)
    with torch.no_grad():
        A = m2.forward_quantized_prefix(x, i - 1, TOTAL_BITS, FRAC_BITS)
    loss = F.cross_entropy(m2.forward_from(A, i), torch.randint(0, 10, (64,)))
    loss.backward()
    grads_tail = all(p.grad is not None and p.grad.abs().sum() > 0 for p in m2.layer_parameters(i))
    grads_head = all(p.grad is None for j in range(i) for p in m2.layers[j].parameters())
    check("backprop from the quantized prefix reaches the trainable tail", grads_tail)
    check("... and never the frozen quantized head", grads_head)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        model.save(path)
        loaded = TorchCNN.load(path)
        check("save/load round trip reproduces outputs exactly", torch.equal(loaded(x), logits))

print("\n=== quantizer and format helpers ===")
# f = total_bits: range +-2^(8-1-8) = +-0.5 with a 2^-8 step
v = torch.tensor([0.3, -0.49, 0.1234])
q = fixed_point_quantize(v, total_bits=8, fractional_bits=8)
check("fractional_bits = total_bits: pure-fraction format, range +-0.5",
      torch.equal(q, torch.round(v * 256) / 256), str(q.tolist()))
q9 = fixed_point_quantize(torch.tensor([0.3, -0.3]), total_bits=8, fractional_bits=9)
check("... and one more fractional bit halves the range to +-0.25 (clips)",
      q9.tolist() == [127 / 512, -0.25], str(q9.tolist()))
check("bits_to_cover", [bits_to_cover(m, 8) for m in (0.3, 1.0, 5.0, 8.0, 8.1)] == [8, 7, 4, 4, 3],
      str([bits_to_cover(m, 8) for m in (0.3, 1.0, 5.0, 8.0, 8.1)]))

print(f"\n{passed} passed, {failed} failed")
