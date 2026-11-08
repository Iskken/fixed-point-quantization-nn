"""
Planned checks (run with: python -m pytest piecewise_nonlinearity/tests):
  - at test time every quantizer gives the same output (exact rounding)
  - EWGS with delta = 0 gives exactly the STE gradients
  - the staircase with alpha = beta = 1 is the identity, and at large alpha it approaches exact rounding
  - the staircase is continuous at the bin edges, and its "exact" backward matches finite differences
  - gradients are zero outside the clipping range
"""
