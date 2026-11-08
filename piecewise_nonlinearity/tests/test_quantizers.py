"""
Planned checks (run with: python -m pytest piecewise_nonlinearity/tests):
  - all quantizers give the same forward output (exact rounding)
  - piecewise hybrid with s1 = s2 = 1, and EWGS with delta = 0, give exactly the STE gradients
  - gradients are zero outside the clipping range
"""
