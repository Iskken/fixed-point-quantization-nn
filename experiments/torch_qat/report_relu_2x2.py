import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import (plot_r22_curves, plot_r22_final, plot_r22_scope, r22_refs,
                                         write_r22_summary)

"""
Rebuild the ReLU 2x2 figures and table from the saved runs. No training.

python -m experiments.torch_qat.report_relu_2x2
"""

DIR = os.environ.get("TWOXTWO_DIR", "results/torch_qat/relu_2x2")


def main():
    raw = os.path.join(DIR, "raw")
    runs = load_all(raw)
    refs = r22_refs(raw)
    print(f"{len(runs)} runs, {len(refs)} starts")
    plot_r22_final(runs, refs, DIR)
    plot_r22_scope(runs, DIR, "test")
    plot_r22_scope(runs, DIR, "val")
    plot_r22_curves(runs, refs, DIR)
    write_r22_summary(runs, refs, DIR)


if __name__ == "__main__":
    main()
