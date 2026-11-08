"""
Our piecewise-linear staircase, usable from any module (plain PyTorch, no other repo imports).

Planned, ported from src/quantization/torch_stair.py:
  geometry: flat width a = alpha*D/(alpha+1), rise c = D/(1+beta*alpha)
  slopes:   flat s1 = (alpha+1)/(alpha(1+beta*alpha)), steep s2 = beta*alpha(alpha+1)/(1+beta*alpha)
  modes:
    "hybrid"    exact rounding forward, s1/s2 backward (the main candidate)
    "staircase" staircase forward and backward (the TDK variant)
  alpha/beta fixed or scheduled; with s1 = s2 = 1 the hybrid must equal the STE bit for bit.
"""
