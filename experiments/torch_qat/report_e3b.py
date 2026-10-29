import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import (plot_e3b_alpha, plot_e3b_curves, plot_e3b_paired,
                                         plot_e3b_steep, write_e3b_summary)

"""
Rebuild the E3b figures and table from the saved runs. No training.
E3a's hard-STE runs are read as flat reference lines (hard STE has no alpha).

python -m experiments.torch_qat.report_e3b
"""

E3B_DIR = os.environ.get("E3B_DIR", "results/torch_qat/e3b")
E3A_DIR = os.environ.get("E3_DIR", "results/torch_qat/e3")


def main():
    e3b = load_all(os.path.join(E3B_DIR, "raw"))
    e3a = load_all(os.path.join(E3A_DIR, "raw"))
    print(f"{len(e3b)} E3b runs, {len(e3a)} E3a runs")
    for start in sorted({r["meta"]["start"] for r in e3b.values()}):
        plot_e3b_alpha(e3b, e3a, start, E3B_DIR)
        plot_e3b_paired(e3b, e3a, start, E3B_DIR)
        plot_e3b_curves(e3b, start, E3B_DIR)
        plot_e3b_steep(e3b, start, E3B_DIR)
        write_e3b_summary(e3b, e3a, start, E3B_DIR)


if __name__ == "__main__":
    main()
