import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import plot_e3_lr, write_e3_summary

"""
Rebuild the E3a figures and tables from the saved runs. No training.

python -m experiments.torch_qat.report_e3          (E3_DIR=... to point elsewhere)
"""

RESULTS_DIR = os.environ.get("E3_DIR", "results/torch_qat/e3")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")


def main():
    runs = load_all(RAW_DIR)
    print(f"{len(runs)} runs")
    for start in sorted({r["meta"]["start"] for r in runs.values()}):
        plot_e3_lr(runs, start, RESULTS_DIR)
        plot_e3_lr(runs, start, RESULTS_DIR, methods=["qat_clamped"], tag="_clamped_only")
        write_e3_summary(runs, start, RESULTS_DIR)


if __name__ == "__main__":
    main()
