import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import (plot_grid_alpha, plot_grid_blowup, plot_grid_curves,
                                         plot_grid_heatmap, plot_grid_lr, write_grid_summary)

"""
Rebuild the ReLU lr x alpha grid figures and table from the saved runs. No training.

python -m experiments.torch_qat.report_relu_grid
"""

DIR = os.environ.get("GRID_DIR", "results/torch_qat/relu_grid")


def main():
    runs = load_all(os.path.join(DIR, "raw"))
    print(f"{len(runs)} runs")
    plot_grid_heatmap(runs, DIR, "test")
    plot_grid_heatmap(runs, DIR, "val")
    plot_grid_lr(runs, DIR)
    plot_grid_alpha(runs, DIR)
    plot_grid_curves(runs, DIR, lr=0.03)
    plot_grid_curves(runs, DIR, lr=0.01)
    plot_grid_blowup(runs, DIR)
    write_grid_summary(runs, DIR)


if __name__ == "__main__":
    main()
