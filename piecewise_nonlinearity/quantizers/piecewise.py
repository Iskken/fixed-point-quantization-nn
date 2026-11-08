"""
Our pure piecewise-linear staircase S_{alpha,beta}, usable from any module (plain PyTorch, no other repo imports).

Planned, ported from src/quantization/torch_stair.py and applied on the shared quantizer's grid (base.py):
  geometry: flat width a = alpha*D/(alpha+1), rise c = D/(1+beta*alpha)      (D = grid step)
  slopes:   flat s1 = (alpha+1)/(alpha(1+beta*alpha)), steep s2 = beta*alpha(alpha+1)/(1+beta*alpha)
  training forward: S(x); test forward: exact rounding, like every other method
  backward switch: "exact" (slopes of S, the main method) or "ste" (slope 1, ablation)
  alpha scheduled from alpha_start to alpha_end; beta fixed per run
  (S is the identity only at alpha = beta = 1; with beta != 1 it is already distorted at alpha = 1)
No hybrid mode: exact rounding forward with s1/s2 backward is STE in disguise under Adam
(Schoenbauer et al. 2024), and in the MLP study it cut oscillation without improving the loss.
Lessons from the MLP study (results/torch_qat/study_claude/STUDY_LOG.md): exact backward with beta = 1
and large alpha freezes the quantized weights; the one win over hard STE was exact, alpha 2, beta 2 (8/4).
"""
