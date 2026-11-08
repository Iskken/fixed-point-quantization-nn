"""
DSQ (Gong et al., ICCV 2019).

Planned:
  phi(x) = s * tanh(k (x - m_i)) on interval P_i,  m_i = l + (i + 0.5) D,  s = 1/(1 - alpha),
  k = log(2/alpha - 1) / D                                  (Eq. 3-8)
  forward: sign of phi gives the hard quantized value (Algorithm 1); backward through phi
  learnable alpha (init 0.2, kept in (0, 0.5), k <= 1000) and clipping bounds l, u.
  Forward checked in the paper's Algorithm 1 (step 3): w_q = sgn(phi(w)), then dequantized, so the training
  forward is hard; only the backward goes through phi.
"""
