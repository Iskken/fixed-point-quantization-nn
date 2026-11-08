"""
Shared quantizer used by every method: the same clipping, the same levels, and the same
test-time forward pass (exact rounding).

Planned (following EWGS Eq. 1-3, which DSQ's setup also matches):
  x_n = clip((x - l) / (u - l), 0, 1)          learnable bounds l, u
  x_q = round((2^b - 1) x_n) / (2^b - 1)        exact rounding (always at test time)
  weights: 2 (x_q - 0.5); activations: x_q (non-negative, after ReLU)
The step from x_n to x_q is supplied by ste.py, ewgs.py and dsq.py (exact rounding forward, own
backward rule) or by piecewise.py (staircase S during training, exact rounding at test time).
A power-of-two fixed-point variant (our previous setting) can be added later as an option.
"""
