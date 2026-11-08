"""
QAT fine-tuning from the float checkpoint, one entry point for all methods:
  python -m piecewise_nonlinearity.train_qat --method {ste,piecewise,ewgs,dsq} --bits W/A --seed N
Writes a per-run JSON summary (config, val/test accuracy, best epoch) to results/.
"""
