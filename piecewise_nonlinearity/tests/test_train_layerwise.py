"""
Checks for train_layerwise.py.

Without data (a randomly initialized quantized ResNet-20 on the CPU):
  - the network splits into the float stem, the 18 quantized convolutions with their BatchNorms, and the head;
  - stage k sets every layer's mode: layers before k frozen (no gradient, eval mode, exact rounding), layer k
    with S, the later layers and the head with the STE; the later layers really are the STE;
  - training steps in stage k change layer k, the later layers and the head, and nothing before layer k
    (weights, quantizer bounds, output scales, BatchNorm parameters and statistics);
  - alpha ramps from alpha_start to alpha_end in every stage; the learning rates of layer k (only) follow alpha;
  - early stopping: ramp epochs never count, ties are no improvement, the patience and the maximum end a stage;
  - run names contain every setting; invalid settings are rejected.
With CIFAR-10 and the float models (smoke runs of 2 batches of 32 images on the CPU):
  - a run trains all 18 stages in order, writes its JSON with aligned curves, removes its checkpoint, and is
    skipped when started again;
  - after every stage the network is back at the stage's best epoch, and a frozen layer never changes again;
  - layers are frozen with BatchNorm statistics re-estimated for the rounded network, so the final
    re-estimation changes nothing; results from before this rule are not reused;
  - alpha restarts in every stage;
  - an interrupted run resumes and ends like an uninterrupted one;
  - the start is the same quantized network as for train_qat.py; seed 0 never evaluates the test set, seed 1 does.
"""

import json
import os
import shutil

import numpy as np
import pytest
import torch

from piecewise_nonlinearity import train_layerwise as lw
from piecewise_nonlinearity import train_qat
from piecewise_nonlinearity.datasets import DATA_DIR
from piecewise_nonlinearity.models.quant_layers import init_quantizers, qconv_factory, quantizers
from piecewise_nonlinearity.models.resnet20 import ResNet20
from piecewise_nonlinearity.quantizers.base import Quantizer
from piecewise_nonlinearity.quantizers.piecewise import Staircase
from piecewise_nonlinearity.quantizers.ste import STE

BASE = ["--group", "smoke", "--bits", "2/2"]


def args(*extra):
    return lw.parse_args([*BASE, *extra])


def tiny_model():
    """A quantized ResNet-20 (2/2, staircase quantizers) with random weights and initialized quantizers."""
    torch.manual_seed(0)
    model = ResNet20(conv_layer=qconv_factory("piecewise", 2, 2, alpha=1.0, beta=2.0, backward="exact"))
    init_quantizers(model, torch.randn(32, 3, 32, 32))
    return model


def random_batches(n=3, size=16):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(size, 3, 32, 32, generator=g), torch.randint(0, 10, (size,), generator=g)) for _ in range(n)]


# --------------------------------------------------------------------------------------- the network and stages

def test_the_network_splits_into_stem_18_quantized_layers_and_head():
    model = tiny_model()
    units, head = lw.network_units(model)
    assert len(units) == 19 and units[0][0] == "stem" and head[0] == "head"
    assert [n for n, _ in units[1:]] == [n for n, _ in train_qat.quantized_convs(model)]
    assert units[1][1] == [model.layers[0].conv1, model.layers[0].bn1]
    assert units[18][1] == [model.layers[8].conv2, model.layers[8].bn2]
    covered = {id(p) for _, mods in units + [head] for m in mods for p in m.parameters()}
    assert covered == {id(p) for p in model.parameters()}  # every parameter belongs to exactly one unit


@pytest.mark.parametrize("k", [1, 7, 18])
def test_a_stage_sets_every_layers_mode(k):
    model = tiny_model()
    stage = lw.Stage(model, k, args(), steps_per_epoch=4)
    units, head = lw.network_units(model)
    for i, (_, mods) in enumerate(units):
        for m in mods:
            assert all(p.requires_grad == (i >= k) for p in m.parameters()), (i, m)
            for q in quantizers(m):
                assert q.force_hard == (i != k)  # only layer k uses S; frozen and later layers round
    assert all(p.requires_grad for m in head[1] for p in m.parameters())
    stage.train_mode()
    for i, (_, mods) in enumerate(units):
        assert all(m.training == (i >= k) for m in mods)  # frozen BatchNorms keep their statistics
    assert all(m.training for m in head[1])
    names = [g["name"] for o in stage.optimizers for g in o.param_groups]
    assert names.count("current") == 2  # layer k's network and quantizer parameters
    with pytest.raises(ValueError):
        lw.Stage(model, 0, args(), steps_per_epoch=4)  # the stem is never a stage


def test_a_layer_after_k_is_the_ste():
    torch.manual_seed(0)
    x = torch.randn(1000) * 2
    stair = Quantizer("weight", 2, Staircase(alpha=3.0, beta=2.0))
    ste = Quantizer("weight", 2, STE())
    stair.force_hard = True  # what a stage sets for the layers after k
    for q in (stair, ste):
        q.initialize(x)
        q.train()
    xs, xe = x.clone().requires_grad_(), x.clone().requires_grad_()
    ys, ye = stair(xs), ste(xe)
    (ys * torch.arange(1000.0)).sum().backward()
    (ye * torch.arange(1000.0)).sum().backward()
    assert torch.equal(ys, ye) and torch.equal(xs.grad, xe.grad)
    assert torch.equal(stair.l.grad, ste.l.grad) and torch.equal(stair.u.grad, ste.u.grad)


def test_training_in_stage_k_changes_layer_k_and_after_and_nothing_before():
    model = tiny_model()
    k = 6
    stage = lw.Stage(model, k, args("--lr", "1e-2", "--lr-q", "1e-2"), steps_per_epoch=3)
    before = lw.snapshot(model)
    lw.train_epoch(model, stage, random_batches(), 1, 3, torch.device("cpu"))
    after = model.state_dict()
    units, head = lw.network_units(model)
    module_name = {id(m): n for n, m in model.named_modules()}
    frozen = tuple(module_name[id(m)] + "." for _, mods in units[:k] for m in mods)  # stem and layers 1 ... k-1
    checked = [name for name in after if name.startswith(frozen)]
    assert len(checked) > 50 and "conv1.weight" in checked and "layers.2.bn1.running_var" in checked
    for name in checked:  # weights, l, u, output scales, BatchNorm parameters and statistics
        assert torch.equal(after[name], before[name]), f"{name} changed although it is frozen"
    for name in ("layers.2.conv2.weight", "layers.2.conv2.wq.u", "layers.2.conv2.output_scale",  # layer k (unit 6)
                 "layers.2.bn2.weight", "layers.2.bn2.running_mean",
                 "layers.5.conv1.weight", "layers.5.conv1.aq.l", "layers.5.bn1.running_var",  # a later layer
                 "linear.weight", "bn2.running_mean"):  # the head
        assert not torch.equal(after[name], before[name]), f"{name} did not change"
    assert units[k][0] == "layers.2.conv2"


def test_alpha_ramps_in_every_stage_and_only_layer_ks_rates_follow_it():
    model = tiny_model()
    stage = lw.Stage(model, 3, args("--alpha-end", "3", "--stage-ramp-epochs", "2", "--lr", "1e-3",
                                    "--lr-q", "1e-5", "--lr-alpha-power", "2"), steps_per_epoch=4)
    assert stage.alpha_at(0) == pytest.approx(1.25) and stage.alpha_at(3) == pytest.approx(2.0)
    assert stage.alpha_at(7) == 3.0 and stage.alpha_at(50) == 3.0  # alpha_end at the last step of the ramp
    stage.set_step(7)
    assert all(q.discretizer.alpha == 3.0 for q in stage.quantizers)
    rates = {(i, g["name"]): g["lr"] for i, o in enumerate(stage.optimizers) for g in o.param_groups}
    assert rates[(0, "current")] == pytest.approx(1e-3 / 9) and rates[(1, "current")] == pytest.approx(1e-5 / 9)
    assert rates[(0, "later")] == 1e-3 and rates[(1, "later")] == 1e-5
    flat = lw.Stage(model, 3, args("--alpha-end", "3", "--stage-ramp-epochs", "0"), steps_per_epoch=4)
    assert flat.alpha_at(0) == 3.0
    assert {g["lr"] for g in flat.optimizers[0].param_groups} == {1e-3}  # no coupling by default


@pytest.mark.parametrize("ramp, patience, maximum, metrics, best, stopped_at", [
    (2, 3, 15, [99.0, 60.0, 61.0, 61.0, 60.5, 60.0, 70.0], 3, 6),  # ramp epoch 1 ignored; a tie is no improvement
    (0, 2, 15, [50.0, 49.0, 51.0, 50.0, 50.0], 3, 5),  # no ramp: tracking from epoch 1
    (1, 3, 4, [50.0, 51.0, 52.0, 53.0, 54.0], 4, 4),  # still improving: the maximum ends the stage
])
def test_early_stopping(ramp, patience, maximum, metrics, best, stopped_at):
    stopper = lw.EarlyStopping(ramp, patience, maximum)
    for epoch, metric in enumerate(metrics, start=1):
        _, stop = stopper.update(epoch, metric)
        if stop:
            break
    assert (stopper.best_epoch, epoch) == (best, stopped_at)
    assert stopper.reason() == ("max_stage_epochs" if stopped_at == maximum and stopper.since < patience else "patience")


def test_run_names_contain_every_setting():
    a = args("--lr", "3e-3", "--alpha-end", "3", "--beta", "3", "--lr-alpha-power", "2")
    assert lw.run_name(a) == "layerwise_w2a2_lr3e-3_lrq1e-5_wd1e-4_a3b3-exact-lrpow2_ramp2-pat3-max15-reest_s0"
    b = args("--seed", "2", "--backward", "ste", "--stage-ramp-epochs", "0", "--patience", "5",
             "--max-stage-epochs", "20", "--stage-metric", "val_acc", "--alpha-start", "2")
    assert lw.run_name(b) == "layerwise_w2a2_lr1e-3_lrq1e-5_wd1e-4_a2b2-ste-from2_ramp0-pat5-max20_s2"
    assert "stop_after" not in lw.run_settings(args("--stop-after", "3"))  # not a setting


@pytest.mark.parametrize("extra", [
    ["--patience", "0"], ["--stage-ramp-epochs", "-1"], ["--stage-ramp-epochs", "5", "--max-stage-epochs", "4"],
    ["--alpha-end", "0"], ["--stage-metric", "loss"], ["--backward", "hybrid"], ["--train-eval-size", "-1"],
])
def test_invalid_arguments_are_rejected(extra):
    with pytest.raises(SystemExit):
        args(*extra)


def test_bits_must_be_valid():
    with pytest.raises(SystemExit):
        lw.parse_args(["--group", "smoke", "--bits", "32/32"])


# ------------------------------------------------------------------------------ smoke runs with CIFAR-10

FLOAT0 = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed0.pt")
FLOAT1 = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed1.pt")
needs_data = pytest.mark.skipif(
    not (os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")) and os.path.exists(FLOAT0)),
    reason="CIFAR-10 or the seed-0 float model is missing")
SMOKE = ["--group", "smoke", "--bits", "2/2", "--seed", "0", "--max-batches", "2", "--batch-size", "32",
         "--workers", "0", "--device", "cpu", "--train-eval-size", "64",
         "--stage-ramp-epochs", "1", "--patience", "1", "--max-stage-epochs", "3"]


def smoke_run(out_dir, *extra):
    """18 stages of 2-3 epochs of 2 batches of 32 images at 2/2 on the CPU; everything is written below out_dir."""
    return lw.main([*SMOKE, "--results-dir", str(out_dir / "results"), "--raw-dir", str(out_dir / "raw"),
                    "--ckpt-dir", str(out_dir / "ckpt"), *extra])


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("layerwise")
    return out, smoke_run(out)


@needs_data
def test_a_run_trains_every_stage_in_order_and_writes_everything(full):
    out, r = full
    assert [s["layer"] for s in r["stages"]] == r["layers"] and len(r["stages"]) == 18
    assert all(2 <= s["epochs"] <= 3 for s in r["stages"])  # ramp 1 + patience 1, at most 3
    assert r["n_epochs"] == sum(s["epochs"] for s in r["stages"])
    h = r["history"]
    assert all(len(v) == r["n_epochs"] + 1 for v in h.values())  # the start and every epoch
    assert h["stage"] == [None] + [s["stage"] for s in r["stages"] for _ in range(s["epochs"])]
    assert r["method"] == "layerwise" and r["config"]["patience"] == 1
    assert all(v is None for k, v in r["final"].items() if k.startswith("test_"))  # seed 0: no test set
    assert all(r["final"][k] is not None for k in ("val_acc", "val_acc_reest", "val_loss", "val_loss_reest"))
    raw = out / "raw" / "smoke" / r["name"]
    assert (raw / "final.pt").exists() and (raw / "steps.npz").exists()
    assert not (out / "ckpt" / "smoke" / f"{r['name']}_resume.pt").exists()
    json_path = out / "results" / "smoke" / f"{r['name']}.json"
    mtime = json_path.stat().st_mtime
    assert smoke_run(out)["final"] == r["final"] and json_path.stat().st_mtime == mtime  # skipped


@needs_data
def test_every_stage_goes_back_to_its_best_epoch_and_frozen_layers_never_change(full):
    _, r = full
    h = r["history"]
    for s in r["stages"]:
        i = next(i for i in range(len(h["epoch"])) if h["stage"][i] == s["stage"] and h["stage_epoch"][i] == s["best_epoch"])
        # the rolled-back network is the best epoch's; the default metric is the one after re-estimation
        assert s["val_acc"] == h["val_acc"][i] and s["val_acc_reest"] == h["val_acc_reest"][i] == s["best_metric"]
        j = s["stage"] - 1  # the layer's index in the per-layer lists
        later = [n for n in range(len(h["epoch"])) if h["stage"][n] is not None and h["stage"][n] > s["stage"]]
        for key in ("w_l", "w_u", "a_l", "a_u", "output_scale"):
            assert {h[key][n][j] for n in later} <= {h[key][i][j]}, (key, s["layer"])


def test_the_default_early_stopping_metric_is_after_batchnorm_re_estimation():
    assert args().stage_metric == "val_acc_reest"  # the supervisor, 2026-11-21


@needs_data
def test_every_epoch_is_validated_after_re_estimation_ramp_epochs_included(tmp_path):
    r = smoke_run(tmp_path, "--stage-ramp-epochs", "2", "--max-stage-epochs", "3")  # epoch 1 of a stage: ramp
    h = r["history"]
    assert [h["tracking"][i] for i in range(1, len(h["epoch"])) if h["stage_epoch"][i] == 1] == [False] * 18
    assert all(v is not None for v in h["val_acc_reest"]) and all(v is not None for v in h["train_eval_acc_reest"])


@needs_data
def test_layers_are_frozen_with_the_batchnorm_statistics_of_the_rounded_network(full):
    """Every BatchNorm is re-estimated with exact rounding before its layer is frozen, so re-estimating the
    deployed network at the end changes nothing: the later layers trained on the statistics that are kept."""
    out, r = full
    saved = torch.load(out / "raw" / "smoke" / r["name"] / "final.pt", weights_only=True)
    reest = saved["bn_reestimated"]
    assert reest and all(k in saved["model"] for k in reest)
    for k, v in reest.items():
        if not k.endswith("num_batches_tracked"):
            assert torch.allclose(saved["model"][k], v, rtol=1e-4, atol=1e-6), k
    assert r["final"]["val_acc"] == r["final"]["val_acc_reest"]
    assert r["frozen_bn"] == lw.FROZEN_BN


@needs_data
def test_a_result_from_before_the_re_estimation_at_freezing_is_not_reused(full, tmp_path):
    out, r = full
    shutil.copytree(out / "results", tmp_path / "results")
    path = tmp_path / "results" / "smoke" / f"{r['name']}.json"
    old = json.loads(path.read_text())
    del old["frozen_bn"]
    path.write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="before BatchNorm re-estimation"):
        smoke_run(tmp_path)


@needs_data
def test_alpha_restarts_in_every_stage(full):
    out, r = full
    steps = np.load(out / "raw" / "smoke" / r["name"] / "steps.npz")
    for s in r["stages"]:
        alphas = steps["alpha"][steps["stage"] == s["stage"]]
        assert np.allclose(alphas, [1.5] + [2.0] * (len(alphas) - 1))  # ramp over epoch 1 (2 steps), then alpha_end
    assert len(steps["loss"]) == 2 * r["n_epochs"]


@needs_data
def test_an_interrupted_run_resumes_and_ends_like_an_uninterrupted_one(full, tmp_path):
    _, r = full
    assert smoke_run(tmp_path, "--stop-after", "4") is None  # inside a stage or at its end, depending on stage 1
    resumed = smoke_run(tmp_path)
    for k in ("val_acc", "val_acc_soft", "stage", "stage_epoch", "alpha", "lr", "w_u"):
        assert resumed["history"][k] == r["history"][k], k
    assert np.allclose(resumed["history"]["train_loss"][1:], r["history"]["train_loss"][1:], rtol=1e-5)
    assert [s["best_epoch"] for s in resumed["stages"]] == [s["best_epoch"] for s in r["stages"]]
    numbers = lambda final: {k: v for k, v in final.items() if v is not None}  # seed 0: the test entries are None
    assert numbers(resumed["final"]) == pytest.approx(numbers(r["final"]), rel=1e-5)
    assert resumed["final"].keys() == r["final"].keys()


@needs_data
def test_the_start_is_the_same_network_as_for_train_qat(full, tmp_path):
    _, r = full
    qat = train_qat.main(["--group", "smoke", "--method", "ste", "--bits", "2/2", "--seed", "0", "--epochs", "1",
                          "--max-batches", "2", "--batch-size", "32", "--workers", "0", "--device", "cpu",
                          "--train-eval-size", "64", "--results-dir", str(tmp_path / "res"),
                          "--raw-dir", str(tmp_path / "raw"), "--ckpt-dir", str(tmp_path / "ckpt")])
    for k in ("val_acc", "val_acc_reest", "train_eval_acc", "w_l", "w_u", "a_l", "a_u", "output_scale"):
        assert r["history"][k][0] == qat["history"][k][0], k


@needs_data
@pytest.mark.skipif(not os.path.exists(FLOAT1), reason="the seed-1 float model is missing")
def test_a_fresh_seed_evaluates_the_test_set(tmp_path):
    r = smoke_run(tmp_path, "--seed", "1", "--max-stage-epochs", "1", "--stage-ramp-epochs", "0")
    assert r["config"]["seed"] == 1 and all(s["epochs"] == 1 for s in r["stages"])
    assert all(r["final"][k] is not None for k in ("test_acc", "test_acc_reest", "test_loss", "test_loss_reest"))
