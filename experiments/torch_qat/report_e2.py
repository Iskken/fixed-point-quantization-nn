import json
import os

from experiments.torch_qat.storage import load_all
from src.visualization.qat_plots import make_e2_figures, make_e2_timeline_figures

"""
Rebuild every E2 figure and the summary table from the saved raw data.
No training happens here.

python -m experiments.torch_qat.report_e2          (E2_DIR=... to point elsewhere)
"""

RESULTS_DIR = os.environ.get("E2_DIR", "results/torch_qat/e2")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")


def main():
    runs = load_all(RAW_DIR)
    qat_runs = {n: r for n, r in runs.items() if "method" in r["meta"]}
    start_info = {}
    for fn in os.listdir(RAW_DIR):
        if fn.endswith("_start.json"):
            with open(os.path.join(RAW_DIR, fn)) as f:
                d = json.load(f)
            start_info[(d["seed"], d["start"])] = d
    print(f"{len(qat_runs)} method runs, {len(start_info)} (seed, start) pairs")
    float_runs = {n: r for n, r in runs.items() if n.endswith("_float")}
    make_e2_figures(qat_runs, start_info, RESULTS_DIR, float_runs=float_runs)
    make_e2_timeline_figures(qat_runs, float_runs, start_info, RESULTS_DIR)


if __name__ == "__main__":
    main()
