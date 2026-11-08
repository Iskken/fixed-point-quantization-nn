"""
Shared quantizer used by every method, so that only the backward rule differs.

Planned (following EWGS Eq. 1-3, which DSQ's setup also matches):
  x_n = clip((x - l) / (u - l), 0, 1)          learnable bounds l, u
  x_q = round((2^b - 1) x_n) / (2^b - 1)        exact rounding in the forward pass
  weights: 2 (x_q - 0.5); activations: x_q (non-negative, after ReLU)
The backward rule from x_q to x_n is supplied by ste.py, piecewise.py, ewgs.py or dsq.py.
A power-of-two fixed-point variant (our previous setting) can be added later as an option.
"""
