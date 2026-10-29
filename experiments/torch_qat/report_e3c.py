import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import (plot_e3c_curves, plot_e3c_final, plot_e3c_schedule,
                                         write_e3c_summary)

"""
Rebuild the E3c figures and table from the saved runs. No training.
E3a's hard-STE runs are read as the reference line.

python -m experiments.torch_qat.report_e3c
"""

E3C_DIR = os.environ.get("E3C_DIR", "results/torch_qat/e3c")
E3A_DIR = os.environ.get("E3_DIR", "results/torch_qat/e3")


def main():
    e3c = load_all(os.path.join(E3C_DIR, "raw"))
    e3a = load_all(os.path.join(E3A_DIR, "raw"))
    print(f"{len(e3c)} E3c runs, {len(e3a)} E3a runs")
    plot_e3c_final(e3c, e3a, E3C_DIR)
    plot_e3c_schedule(e3c, E3C_DIR)
    plot_e3c_curves(e3c, E3C_DIR)
    write_e3c_summary(e3c, e3a, E3C_DIR)


if __name__ == "__main__":
    main()
