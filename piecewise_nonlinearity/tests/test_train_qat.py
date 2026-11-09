"""
Smoke tests for train_qat.py on the CPU: 2 epochs of 2 batches of real CIFAR-10 images (skipped when the
data or the seed-0 float model is missing):
  - a run writes its JSON with every curve, its final weights and its per-step log, and removes its
    resume checkpoint; the same arguments then skip the run;
  - the STE and the staircase with the same seed start from the same quantized network;
  - alpha rises linearly and reaches alpha_end at half of the steps;
  - an interrupted run resumes from its checkpoint and ends like an uninterrupted one;
  - run names contain every setting, and invalid settings are rejected.
"""

import os

import numpy as np
import pytest

from piecewise_nonlinearity import train_qat
from piecewise_nonlinearity.datasets import DATA_DIR

FLOAT_MODEL = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed0.pt")
needs_data = pytest.mark.skipif(
    not (os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")) and os.path.exists(FLOAT_MODEL)),
    reason="CIFAR-10 or the seed-0 float model is missing")


def smoke_run(out_dir, *args):
    """2 epochs of 2 batches of 32 images at 2/2 on the CPU; everything is written below out_dir."""
    return train_qat.main(["--group", "smoke", "--bits", "2/2", "--seed", "0", "--epochs", "2",
                           "--max-batches", "2", "--batch-size", "32", "--workers", "0", "--device", "cpu",
                           "--reestimate-every", "1", "--save-every", "1",
                           "--results-dir", str(out_dir / "results"), "--raw-dir", str(out_dir / "raw"),
                           "--ckpt-dir", str(out_dir / "ckpt"), *args])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("qat")
    return out, {m: smoke_run(out, "--method", m) for m in ("ste", "piecewise")}


@needs_data
def test_run_writes_everything_and_is_skipped_when_done(runs):
    out, results = runs
    r = results["piecewise"]
    json_path = out / "results" / "smoke" / f"{r['name']}.json"
    raw = out / "raw" / "smoke" / r["name"]
    assert json_path.exists() and (raw / "final.pt").exists()
    assert not (out / "ckpt" / "smoke" / f"{r['name']}_resume.pt").exists()
    steps = np.load(raw / "steps.npz")
    assert len(steps["loss"]) == len(steps["correct"]) == len(steps["alpha"]) == 4
    assert all(len(v) == 3 for v in r["history"].values())  # the start and 2 epochs
    assert r["history"]["val_acc_reest"][1] is not None  # --reestimate-every 1
    for k in ("val_acc_reest", "test_acc_reest", "val_acc", "test_acc", "val_acc_soft", "test_acc_soft"):
        assert r["final"][k] is not None
    assert not r["failed"] or r["failure"].startswith("ended below its start")  # 4 steps may not beat the start
    mtime = json_path.stat().st_mtime
    assert smoke_run(out, "--method", "piecewise")["final"] == r["final"]
    assert json_path.stat().st_mtime == mtime


@needs_data
def test_methods_with_the_same_seed_start_from_the_same_network(runs):
    _, results = runs
    ste, stair = results["ste"]["history"], results["piecewise"]["history"]
    for k in ("w_l", "w_u", "a_l", "a_u", "output_scale", "val_acc", "val_acc_reest"):
        assert ste[k][0] == stair[k][0]
    assert "alpha" not in ste and results["ste"]["final"].get("val_acc_soft") is None


@needs_data
def test_alpha_rises_linearly_to_alpha_end_at_half_of_the_steps(runs):
    out, results = runs
    r = results["piecewise"]
    steps = np.load(out / "raw" / "smoke" / r["name"] / "steps.npz")
    assert np.allclose(steps["alpha"], [1.0, 1.5, 2.0, 2.0])
    assert r["history"]["alpha"] == [1.0, 1.5, 2.0]  # after each epoch's last step


@needs_data
def test_interrupted_run_resumes_and_ends_like_an_uninterrupted_one(runs, tmp_path):
    _, results = runs
    assert smoke_run(tmp_path, "--method", "piecewise", "--stop-after", "1") is None
    resumed = smoke_run(tmp_path, "--method", "piecewise")
    full = results["piecewise"]
    assert np.allclose(resumed["history"]["train_loss"][1:], full["history"]["train_loss"][1:], rtol=1e-5)
    assert resumed["history"]["val_acc"] == full["history"]["val_acc"]
    assert resumed["final"] == pytest.approx(full["final"], rel=1e-5)


def test_run_names_contain_every_setting():
    a = train_qat.parse_args(["--group", "tune", "--method", "piecewise", "--bits", "2/2", "--lr", "3e-4",
                              "--alpha-end", "5", "--beta", "1"])
    assert train_qat.run_name(a) == "piecewise_w2a2_lr3e-4_lrq1e-5_wd1e-4_a5b1-exact-r0.5_e60_s0"
    b = train_qat.parse_args(["--group", "tune", "--method", "ste", "--bits", "4/32", "--seed", "3",
                              "--weight-decay", "0"])
    assert train_qat.run_name(b) == "ste_w4a32_lr1e-3_lrq1e-5_wd0_e60_s3"
    assert (b.w_bits, b.a_bits) == (4, None)


@pytest.mark.parametrize("argv", [
    ["--method", "ste", "--bits", "2/2", "--alpha-end", "3"],  # a staircase setting for the STE
    ["--method", "ewgs", "--bits", "2/2"],  # added in step 8
    ["--method", "ste", "--bits", "32/32"],  # nothing quantized
    ["--method", "ste", "--bits", "2"],
    ["--method", "piecewise", "--bits", "2/2", "--ramp-fraction", "0"],
])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        train_qat.parse_args(["--group", "tune", *argv])
