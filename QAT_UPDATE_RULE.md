# How the weight update works in stair QAT

This note explains how a weight is updated during quantization-aware training
(QAT) with the tunable stair non-linearity, why the update contains a "slope"
factor, and how the three gradient modes (`exact`, `clamped`, `ste`) differ.

Code this note refers to:

| What | Where |
|---|---|
| Stair function, forward and backward | [src/quantization/torch_stair.py](src/quantization/torch_stair.py) (`StairQuantize`, `stair_quantize`) |
| QAT forward pass and training loop | [src/models/torch_qat.py](src/models/torch_qat.py) (`QATMLP.forward_qat`, `train_qat`) |
| NumPy reference implementation | [src/models/mlp.py](src/models/mlp.py) (`_stair_quantize`, `backward_qat_non_linear`) |
| E1 experiment (variants below) | [experiments/torch_qat/run_e1_baselines.py](experiments/torch_qat/run_e1_baselines.py) |

---

## 1. Two copies of every weight

QAT keeps two versions of each weight and bias:

| | Master weight `W` | Quantized weight `Wq` |
|---|---|---|
| Type | ordinary float | `Wq = stair(W)` |
| Who changes it | the optimizer | nobody; recomputed from `W` every forward pass |
| Who uses it | nothing in the forward pass | the layer: `z = a @ Wq + bq` |

The network computes with `Wq`, but the optimizer only ever changes `W`. The
stair function sits between them, and that is why the update has an extra
factor compared with float training.

## 2. One training step

For one quantized weight `w` (a single entry of `W`), with quantized value
`wq = stair(w)`:

```
1. forward     wq = stair(w)                  # stair uses the current alpha, beta
               loss = L( network using wq )

2. backward    g_q = dL/d(wq)                 # same computation as in float training

3. backward    slope = d(wq)/dw               # the stair's local slope at w
               g   = g_q * slope              # chain rule: dL/dw = dL/dwq * dwq/dw

4. update      w  <-  w - lr * g              # applied to the master float weight
```

Steps 2 and 3 are one application of the chain rule. The loss depends on `w`
only through `wq`, so

```
dL/dw = dL/dwq * dwq/dw
        ^^^^^^   ^^^^^^
        as in    the "slope": how far does the quantized weight
        float    move when the master weight moves a little?
```

In code, `train_qat` calls `loss.backward()`. PyTorch runs
`StairQuantize.backward`, which multiplies the incoming gradient by
`local_grad` (step 3). Then `optimizer.step()` does plain SGD on `W` (step 4).

## 3. What "slope" means

The slope is the local derivative of the stair output with respect to its
input: how many units the output moves per unit the input moves.

During training it works as a **gain** on each weight's gradient:

- slope 1: the gradient passes through unchanged
- slope 0.1: the gradient is shrunk 10x, so that weight moves 10x less
- slope 10: the gradient is amplified 10x

So a weight's effective learning rate is `lr * slope`, and it changes as the
weight moves along the stair.

### Why a slope is needed at all

A hard quantizer, `round(w)`, is flat between its jumps. Its derivative is 0
almost everywhere, so `dL/dw = 0` and no weight would ever change. The stair
replaces the hard step with a piecewise-linear curve that has a non-zero slope
everywhere, which gives the optimizer a signal.

## 4. The stair, and where its slopes come from

`lsb = 2^-frac_bits` is the spacing of the fixed-point grid (0.0625 for 4
fractional bits). Around every grid point the stair has a **flat** part and a
**steep** part:

```
widths   a = alpha*lsb/(alpha+1)      b = lsb/(alpha+1)           a + b = lsb
heights  c = lsb/(1+beta*alpha)       d = lsb - c                 c + d = lsb

flat  zone: |w - grid point| <= a/2          slope = c/a
steep zone: a/2 < |w - grid point| <= lsb/2  slope = d/b
```

For `beta = 1` these simplify to **flat slope = 1/alpha** and **steep slope =
alpha**. For general beta:

```
flat  slope = (alpha+1) / (alpha*(1+beta*alpha))
steep slope = beta*alpha*(alpha+1) / (1+beta*alpha)
```

Things to know about the shape:

- The curve passes exactly through every grid point and is continuous.
- `alpha = beta = 1` is the identity (slope 1 everywhere).
- As alpha grows the flat zone takes up a larger share of each step
  (50% at alpha=1, 91% at alpha=10, 99% at alpha=100), and the steep zone
  becomes narrower and steeper.
- For alpha < 1 the picture inverts (steep zone dominates). The stair in the
  original design drawing corresponds to alpha > 1.

## 5. The three gradient modes

The **forward pass is identical in all modes.** Only the slope used in step 3
differs. `grad_mode` is an argument of `stair_quantize`.

| mode | slope used in backward |
|---|---|
| `exact` | the true derivative of the forward: `c/a` in the flat zone, `d/b` in the steep zone |
| `clamped` | the NumPy surrogate: flat slope clipped to [0.1, 1], steep slope clipped to [0, 1] |
| `ste` | straight-through estimator: slope 1 everywhere inside the representable range |

### Numbers (beta = 1), computed from `stair_geometry`

| alpha | flat zone share of a step | true slope (flat / steep) | clamped (flat / steep) | STE (flat / steep) |
|---|---|---|---|---|
| 0.1 | 9% | 10 / 0.1 | 1.0 / 0.1 | 1 / 1 |
| 1 | 50% | 1 / 1 | 1 / 1 | 1 / 1 |
| 2 | 67% | 0.5 / 2 | 0.5 / 1 | 1 / 1 |
| 5 | 83% | 0.2 / 5 | 0.2 / 1 | 1 / 1 |
| 10 | 91% | 0.1 / 10 | 0.1 / 1 | 1 / 1 |
| 100 | 99% | 0.01 / 100 | 0.1 / 1 | 1 / 1 |

`exact` uses the "true slope" column. For alpha <= 10 it equals `clamped` in
the flat zone, and the two differ only in the steep zone, where `clamped` caps
the slope at 1.

### How they behave

- **`exact`** reports the truth. Weights in the steep zone get gradients up to
  alpha times larger and can overshoot; weights in the flat zone barely move.
  At alpha=100 the slopes span 0.01 to 100.
- **`clamped`** never amplifies (steep slope capped at 1) and never shrinks a
  flat weight below 0.1. It is bounded, but for alpha > 10 it under-reports the
  steep zone (1 vs alpha) and over-reports the flat zone (0.1 vs 1/alpha).
- **`ste`** ignores the stair shape. It under-reports the steep zone like
  `clamped`, and over-reports the flat zone, which is most of the range at high
  alpha, by a factor of alpha.

### The `legacy` variant in E1

`legacy` is the original NumPy setup: `clamped` slopes, **no saturation mask**,
and alpha ramping from 0.1 instead of 1.0. It differs from `clamped` in two
ways (mask and starting alpha), not in the slope.

## 6. Saturation masking (`mask_saturation`)

The 8-bit format represents only -128*lsb to 127*lsb (-8 to +7.9375 for
4 fractional bits). The forward pass clips anything outside that range. A
clipped output does not change when its input changes, so its true derivative
is 0.

- `mask_saturation=True` (default in the torch code) multiplies the slope by 0
  wherever the forward output was clipped. Forward and backward agree.
- `mask_saturation=False` passes the normal slope even for clipped values. This
  is what the NumPy code does.

It is a separate knob from the slope mode. E1 did not measure how often weights
saturate; with Xavier-scale weights it is expected to be rare.

## 7. Worked example

alpha = 10, lr = 0.01, beta = 1, and the gradient reaching the quantized
weight is `dL/dwq = 2.0`:

| the weight sits in | slope used | `dL/dw` | master weight changes by |
|---|---|---|---|
| flat zone, `exact` or `clamped` | 0.1 | 0.2 | -0.002 |
| flat zone, `ste` | 1 | 2 | -0.02 |
| steep zone, `exact` | 10 | 20 | -0.2 (more than 3 lsb) |
| steep zone, `clamped` or `ste` | 1 | 2 | -0.02 |
| float training (no quantizer) | 1 | 2 | -0.02 |

The steep-zone `exact` step (0.2) is wider than a whole stair step (0.0625).
This is the overshoot risk of the true derivative at large alpha.

The forward effect of a small nudge matches the true slope. Nudging a weight by
0.001 moves the quantized weight by 0.0001 in the flat zone and 0.010 in the
steep zone (alpha = 10).

## 8. Difference from floating-point training

| | Float training | Stair QAT |
|---|---|---|
| Weight used in forward | `W` | `Wq = stair(W)` |
| Gradient of the loss | `dL/dW` directly | `dL/dWq * slope(W)` |
| Effective learning rate | `lr` for all weights | `lr * slope`, different per weight |
| Optimizer step | `W <- W - lr * g` | `W <- W - lr * g` (same rule, on the master weight) |

The update rule itself does not change. What changes is that each weight's
gradient is rescaled by its own position on the stair.

## 9. Where the slope applies

- **Weights and biases:** always, in every QAT run so far.
- **Inputs, pre-activations, activations, output:** only when
  `quantize_activations=True`. It was `False` in E1. When enabled, the stair
  slope also multiplies the gradient flowing back through each quantized
  activation.
- **Not in float training**, and not in plain PTQ, which has no gradient step.

In a hidden layer the stair slope is a separate factor from the activation
derivative (for tanh, `1 - a^2`). Both appear in the backward pass.

## 10. Why do this

Float training ignores rounding, so rounding the trained weights afterwards
(PTQ) can hurt. In E1, mean test MSE was 0.152 for float inference and 0.335
for PTQ of the same models (exact fixed-point). QAT trains through the stair so
the weights settle where rounding costs little. Raising alpha pushes the
training-time function toward the real hard quantizer, and the slope is what
lets the optimizer receive a usable signal through it.

## 11. Status and caveats

- The torch stair is checked against the NumPy implementation
  (`experiments/test_torch_stair.py`, `experiments/test_torch_qat.py`):
  forward, clamped-mode gradients, and 300 epochs of training agree to ~1e-16.
  `exact` mode passes `gradcheck`.
- Which slope mode works best is **not yet established.** In E1 (QAT from the
  same random init as the float model, 5 seeds) the `legacy`, `clamped` and
  `exact` runs all stayed on the loss plateau (test MSE about 0.56) and could
  not be told apart. One STE seed escaped the plateau and four did not. That is
  too little to conclude anything.
- The next experiment (E2) starts QAT from the trained float weights, which
  takes the plateau out of the comparison.
