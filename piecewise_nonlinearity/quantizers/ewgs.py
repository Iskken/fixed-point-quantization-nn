"""
EWGS (Lee et al., CVPR 2021).

Planned:
  backward (Eq. 4): g_xn = g_xq * (1 + delta * sign(g_xq) * (x_n - x_q)), delta >= 0
  delta per quantizer (Eq. 8-10): delta = (Tr(H)/N) / (3 * std(g_xq)), Tr(H) by Hutchinson's method,
  start at 0, update every 10 epochs on CIFAR-10; option for a fixed delta (paper uses 0.01 with DoReFa).
  With delta = 0 it must equal the STE.
"""
