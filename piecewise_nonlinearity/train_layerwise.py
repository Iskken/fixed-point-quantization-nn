"""
Layer-by-layer fine-tuning with the staircase (suggested by D. Hadházi; diagram:
reports/figures/layerwise_staircase_method.png). One run per call, with the same data, starting network and
final evaluation as train_qat.py, so its results compare directly with the other methods.

    python -m piecewise_nonlinearity.train_layerwise --group explore --bits 2/2 --seed 0 \
        --lr 3e-3 --alpha-end 3 --beta 3 --lr-alpha-power 2

The 18 quantized block convolutions are trained one after another, from the input to the output, in 18
stages. In stage k (k = 1 ... 18):
  - layers 1 ... k-1 are frozen at exact rounding: their weights, quantizer bounds, output scales and
    BatchNorms (parameters and running statistics; eval mode) no longer change. Their running statistics are
    those of the rounded network, re-estimated over the full training split just before the layer was frozen;
  - layer k is trained with the staircase: forward S_{alpha,beta}, backward the exact slopes of S
    (--backward ste: slope 1, for an ablation). alpha starts at --alpha-start in every stage, rises linearly
    to --alpha-end over the first --stage-ramp-epochs epochs of the stage, and then stays there;
  - layers k+1 ... 18 are trained with the STE (exact rounding forward, slope 1 backward), and so is the
    float head (the last BatchNorm and the linear layer).
Every stage ends by early stopping with rollback. After every epoch the network is validated the way it will
be deployed: exact rounding in every layer, with BatchNorm re-estimated for that rounded network, on the
validation split. The weights of the best epoch are kept; once --patience epochs in a row bring no
improvement, or after --max-stage-epochs epochs, the network goes back to the best epoch, every BatchNorm is
re-estimated with exact rounding over the full training split, and layer k is frozen with these statistics.
During the stage, layer k's BatchNorm collected statistics of S, not of rounding; freezing them would leave a
mismatch that the later layers adapt to and that the final re-estimation then undoes. The stem is frozen from
the start, so the same re-estimation is done once before stage 1. The epochs of the alpha ramp are never chosen
and do not count towards the patience: tracking starts with the epoch that ends the ramp. After stage 18 every
layer rounds exactly, which is the deployed network, and its statistics are already those of that network.

Decided with the supervisor (2026-11-19): frozen layers round exactly; the optimized layer uses the staircase
with the exact backward; the layers after it use the STE; no fixed number of epochs per stage, but early
stopping with rollback. The choices made here where that left room (the open questions in the diagram):
  1. alpha: a ramp in every stage, over --stage-ramp-epochs epochs (default 2; 0: alpha_end from the start).
  2. Early-stopping metric: validation accuracy with exact rounding after BatchNorm re-estimation
     (--stage-metric val_acc_reest, the default), or with the statistics as trained (val_acc; those were
     collected with S in layer k). Either way, every epoch, the ramp epochs included, is validated both without
     and after re-estimation, as the supervisor asked on 2026-11-21 (about 6 s more per epoch).
  3. --patience (default 3) and --max-stage-epochs (default 15). Every stage gets new Adam optimizers at the
     constant rates --lr and --lr-q, with no decay: early stopping ends the stage. With --lr-alpha-power p,
     the rates of layer k (not those of the later layers) are multiplied by (alpha / alpha_start)^-p at every
     step: only layer k's gradients pass through the slopes of S.
  4. Frozen means everything: weights, l, u, output scale, BatchNorm parameters and statistics, the statistics
     re-estimated with exact rounding over the full training split first (the supervisor, 2026-11-21; not a
     setting).
  5. The float first convolution comes before layer 1, so it stays frozen at the float model's weights; the
     float head comes after layer 18, so it is trained in every stage (not a setting).
  6. The comparison with the STE (120 epochs, no early stopping) is left to the analysis: every result
     records its number of epochs and its run time.
Unlike train_qat.py, which keeps the last epoch, this method selects on the validation set (the rollback), so
its validation accuracy is optimistic; the test set is still used only after the last stage, and only for
fresh seeds (seed >= 1, PLAN.md change 3).

Logged per epoch: the stage, its layer and epoch, alpha, the learning rates, the training loss and accuracy,
the validation accuracy with exact rounding and with S in layer k, the same on --train-eval-size fixed
training images, the early-stopping metric with the best value so far and the epochs since, the share of
weights that changed level, l, u and the output scale of every layer, and the time. Per stage: its epochs,
the chosen epoch, why it stopped, and the validation accuracy after the rollback, with the statistics as
trained (val_acc) and after the re-estimation that the layer is frozen with (val_acc_reest).

Outputs, as for train_qat.py; <name> contains every setting, for example
layerwise_w2a2_lr3e-3_lrq1e-5_wd1e-4_a3b3-exact-lrpow2_ramp2-pat3-max15_s0:
  results/qat/<group>/<name>.json            config, start, stages, per-epoch curves, final metrics (committed)
  results/raw/qat/<group>/<name>/final.pt    final weights and re-estimated BatchNorm statistics (local)
  results/raw/qat/<group>/<name>/steps.npz   loss, correct predictions, alpha and stage at every step (local)
  checkpoints/qat/<group>/<name>_resume.pt   resume checkpoint, saved after every epoch (local)
A run whose JSON exists with the same settings is skipped. An interrupted run continues after its last
finished epoch when started again with the same arguments (on the GPU not bit-identical, as in train_qat.py).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from piecewise_nonlinearity import train_qat
from piecewise_nonlinearity.datasets import (DATASETS, get_bn_loader, get_loaders, get_train_eval_loader,
                                             split_fingerprint)
from piecewise_nonlinearity.models.quant_layers import (QConv2d, init_quantizers, load_float_weights,
                                                         qconv_factory, quantizers)
from piecewise_nonlinearity.models.resnet20 import ResNet20
from piecewise_nonlinearity.quantizers.piecewise import BACKWARDS
from piecewise_nonlinearity.train_qat import (batches, evaluate, file_hash, fmt, git_state, level_changes,
                                              quantized_convs, quantizer_values, reestimated, rounded, short,
                                              weight_levels, write_json)

HERE = train_qat.HERE
METHOD = "layerwise"
METRICS = ("val_acc", "val_acc_reest")
# How a layer's BatchNorm statistics are frozen; stored in every result and resume checkpoint. Runs made before
# 2026-11-22 froze the statistics collected during training (with S) and have no such entry: they are not reused.
FROZEN_BN = "reestimated"
NOT_SETTINGS = train_qat.NOT_SETTINGS  # paths, machine, checkpointing, tests, the training-data measurement


# ----------------------------------------------------------------------------------------------- arguments

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", required=True, help="W/A, e.g. 2/2 or 4/4; 32 = float, so 2/32 is weights only")
    p.add_argument("--seed", type=int, default=0, help="QAT seed, and the float model the run starts from")
    p.add_argument("--group", required=True, choices=train_qat.GROUPS, help="subfolder of the results")
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the network (Adam), constant in a stage")
    p.add_argument("--lr-q", type=float, default=1e-5, help="learning rate of l, u and the output scales (Adam)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="on the network parameters only")
    p.add_argument("--batch-size", type=int, default=256)
    s = p.add_argument_group("the staircase of the optimized layer")
    s.add_argument("--alpha-end", type=float, default=2.0, help="alpha after the ramp of every stage")
    s.add_argument("--beta", type=float, default=2.0, help="fixed for the whole run")
    s.add_argument("--alpha-start", type=float, default=1.0, help="alpha at the first step of every stage")
    s.add_argument("--backward", choices=BACKWARDS, default="exact",
                   help="exact: the slopes of S (the method); ste: slope 1 (ablation)")
    s.add_argument("--lr-alpha-power", type=float, default=0.0,
                   help="the learning rates of the optimized layer follow alpha: multiplied by "
                        "(alpha / alpha_start)^-p at every step (default 0: they do not)")
    e = p.add_argument_group("stages: alpha ramp and early stopping with rollback")
    e.add_argument("--stage-ramp-epochs", type=int, default=2,
                   help="epochs over which alpha rises in every stage (0: alpha_end from the start)")
    e.add_argument("--patience", type=int, default=3,
                   help="a stage ends after this many epochs in a row without a better validation accuracy")
    e.add_argument("--max-stage-epochs", type=int, default=15, help="a stage ends after this many epochs at most")
    e.add_argument("--stage-metric", choices=METRICS, default="val_acc_reest",
                   help="validation accuracy with exact rounding, after (val_acc_reest) or without (val_acc) "
                        "BatchNorm re-estimation")
    p.add_argument("--train-eval-size", type=int, default=5000,
                   help="fixed training images, without augmentation, evaluated every epoch like validation; 0: none")
    p.add_argument("--max-batches", type=int, help="smoke tests: cut every epoch and evaluation to this many batches")
    p.add_argument("--stop-after", type=int, help="tests: stop after this many epochs in total, as if interrupted")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default=os.path.join(HERE, "results", "qat"))
    p.add_argument("--raw-dir", default=os.path.join(HERE, "results", "raw", "qat"))
    p.add_argument("--ckpt-dir", default=os.path.join(HERE, "checkpoints", "qat"))
    p.add_argument("--pretrained-dir", default=os.path.join(HERE, "pretrained"))
    args = p.parse_args(argv)

    try:
        args.w_bits, args.a_bits = train_qat.parse_bits(args.bits)
    except argparse.ArgumentTypeError as err:
        p.error(str(err))
    if args.stage_ramp_epochs < 0:
        p.error("--stage-ramp-epochs must be 0 or more")
    if args.patience < 1:
        p.error("--patience must be at least 1")
    if args.max_stage_epochs < max(args.stage_ramp_epochs, 1):
        p.error("--max-stage-epochs must be at least --stage-ramp-epochs (and at least 1): the ramp must end")
    if min(args.alpha_start, args.alpha_end, args.beta) <= 0:
        p.error("--alpha-start, --alpha-end and --beta must be positive")
    if args.train_eval_size < 0:
        p.error("--train-eval-size must be 0 or more")
    return args


def run_settings(args):
    """What the run's result depends on: every argument except NOT_SETTINGS."""
    return {k: v for k, v in vars(args).items() if k not in NOT_SETTINGS}


def run_name(args):
    """Every setting of the run, so two different runs never share a name."""
    w, a = args.bits.split("/")
    stair = f"a{short(args.alpha_end)}b{short(args.beta)}-{args.backward}"
    if args.alpha_start != 1.0:
        stair += f"-from{short(args.alpha_start)}"
    if args.lr_alpha_power:
        stair += f"-lrpow{short(args.lr_alpha_power)}"
    stop = f"ramp{args.stage_ramp_epochs}-pat{args.patience}-max{args.max_stage_epochs}"
    if args.stage_metric != "val_acc":
        stop += "-reest"
    parts = [METHOD, f"w{w}a{a}", f"lr{short(args.lr)}", f"lrq{short(args.lr_q)}", f"wd{short(args.weight_decay)}",
             stair, stop, f"s{args.seed}"]
    if args.batch_size != 256:
        parts.append(f"bs{args.batch_size}")
    if args.max_batches is not None:
        parts.append(f"mb{args.max_batches}")
    name = "_".join(parts)
    return name if args.dataset == "cifar10" else f"{args.dataset}_{name}"


# ------------------------------------------------------------------------------------------ the network

def build_model(args, device):
    """The quantized ResNet-20 with the float weights of the run's seed, every block convolution with a staircase
    quantizer (a stage switches each one to S, the STE or frozen). Returns (model, float model path)."""
    path = os.path.join(args.pretrained_dir, f"resnet20_{args.dataset}_seed{args.seed}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; train it first with "
                         f"python -m piecewise_nonlinearity.train_float --seed {args.seed}")
    conv = qconv_factory("piecewise", args.w_bits, args.a_bits, alpha=args.alpha_start, beta=args.beta,
                         backward=args.backward)
    model = ResNet20(num_classes=DATASETS[args.dataset]["n_classes"], conv_layer=conv)
    load_float_weights(model, torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.to(device), path


def network_units(model):
    """The network in forward order: the float stem, then the 18 quantized block convolutions, each with the
    BatchNorm right after it ([(name, [modules])], names as in quantized_convs), and the float head."""
    units = [("stem", [model.conv1, model.bn1])]
    for i, block in enumerate(model.layers):
        units += [(f"layers.{i}.conv1", [block.conv1, block.bn1]), (f"layers.{i}.conv2", [block.conv2, block.bn2])]
    return units, ("head", [model.bn2, model.linear])


def split_params(modules):
    """(network parameters, quantizer parameters: l, u and the output scales) of these modules."""
    params = [p for m in modules for p in m.parameters()]
    q_ids = {id(p) for m in modules for q in quantizers(m) for p in q.parameters()}
    q_ids |= {id(m.output_scale) for m in modules if isinstance(m, QConv2d)}
    return [p for p in params if id(p) not in q_ids], [p for p in params if id(p) in q_ids]


def freeze_all(model):
    """The deployed network: every quantizer rounds exactly and nothing is trained any more."""
    for q in quantizers(model):
        q.force_hard = True
    for p in model.parameters():
        p.requires_grad_(False)


class Stage:
    """Stage k (1 ... 18): layers before k frozen at exact rounding, layer k trained with the staircase, the
    layers after it and the float head trained with the STE. Creating it sets every layer's mode and makes new
    Adam optimizers with one group for layer k ("current") and one for the rest ("later")."""

    def __init__(self, model, k, args, steps_per_epoch):
        units, head = network_units(model)
        if not 1 <= k < len(units):
            raise ValueError(f"stage {k} does not exist; the stages are 1 ... {len(units) - 1}")
        self.model, self.k = model, k
        self.layer, (self.conv, self.bn) = units[k][0], units[k][1]
        self.frozen = [m for _, mods in units[:k] for m in mods]
        later = [m for _, mods in units[k + 1:] for m in mods] + head[1]
        for m in self.frozen:
            for p in m.parameters():
                p.requires_grad_(False)
        for m in [self.conv, self.bn] + later:
            for p in m.parameters():
                p.requires_grad_(True)
        for q in quantizers(model):  # frozen layers round; the later ones round too, and as the STE they train
            q.force_hard = True
        self.quantizers = [q for q in (self.conv.wq, self.conv.aq) if q is not None]
        for q in self.quantizers:  # layer k: S in training, exact rounding in eval mode
            q.force_hard = False
        current_net, current_q = split_params([self.conv, self.bn])
        later_net, later_q = split_params(later)

        def groups(current, rest, lr):
            return [{"params": ps, "name": name, "base_lr": lr, "lr": lr}
                    for name, ps in (("current", current), ("later", rest)) if ps]

        self.optimizers = (torch.optim.Adam(groups(current_net, later_net, args.lr), lr=args.lr,
                                            weight_decay=args.weight_decay),
                           torch.optim.Adam(groups(current_q, later_q, args.lr_q), lr=args.lr_q))
        self.alpha_start, self.alpha_end, self.power = args.alpha_start, args.alpha_end, args.lr_alpha_power
        self.ramp_steps = args.stage_ramp_epochs * steps_per_epoch
        self.alpha = None
        self.set_step(0)

    def alpha_at(self, step):
        """alpha at training step `step` of the stage (from 0): linear from alpha_start, reaching alpha_end at
        the last step of the ramp, constant after it."""
        if self.ramp_steps == 0:
            return self.alpha_end
        return self.alpha_start + (self.alpha_end - self.alpha_start) * min(1.0, (step + 1) / self.ramp_steps)

    def set_step(self, step):
        """Before training step `step`: layer k's alpha, and the learning rates (layer k's follow alpha)."""
        self.alpha = self.alpha_at(step)
        for q in self.quantizers:
            q.discretizer.alpha = self.alpha
        scale = 1.0 if not self.power else (self.alpha / self.alpha_start) ** -self.power
        for o in self.optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * (scale if g["name"] == "current" else 1.0)

    def train_mode(self):
        """Train mode for the trained layers; the frozen ones stay in eval mode (BatchNorm statistics fixed)."""
        self.model.train()
        for m in self.frozen:
            m.eval()

    def lrs(self):
        """The rates of the last step: layer k's (network, quantizers) and the later layers' network rate."""
        rate = {(o_i, g["name"]): g["lr"] for o_i, o in enumerate(self.optimizers) for g in o.param_groups}
        return {"lr": rounded(rate[(0, "current")]), "lr_q": rounded(rate.get((1, "current"), float("nan"))),
                "lr_later": rounded(rate[(0, "later")]) if (0, "later") in rate else None}


def train_epoch(model, stage, loader, epoch, steps_per_epoch, device, max_batches=None):
    """One epoch of stage training (epoch counts from 1 in every stage). Returns per-step lists of the loss, the
    number of correct predictions, the batch size and alpha."""
    stage.train_mode()
    losses, corrects, sizes, alphas = [], [], [], []  # kept on the device: no synchronization inside the epoch
    for i, (x, y) in enumerate(batches(loader, max_batches)):
        stage.set_step((epoch - 1) * steps_per_epoch + i)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        for o in stage.optimizers:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in stage.optimizers:
            o.step()
        losses.append(loss.detach())
        corrects.append((out.argmax(1) == y).sum())
        sizes.append(y.numel())
        alphas.append(stage.alpha)
    return torch.stack(losses).tolist(), torch.stack(corrects).tolist(), sizes, alphas


class EarlyStopping:
    """The best epoch and the patience within one stage. The epochs of the alpha ramp are never the best and do
    not count towards the patience: tracking starts with the epoch that ends the ramp (epoch 1 without a ramp).
    A tie is no improvement, so the earlier epoch is kept."""

    def __init__(self, ramp_epochs, patience, max_epochs):
        self.first_tracked, self.patience, self.max_epochs = max(ramp_epochs, 1), patience, max_epochs
        self.best_metric = self.best_epoch = None
        self.since = 0

    def tracking(self, epoch):
        return epoch >= self.first_tracked

    def update(self, epoch, metric):
        """After epoch `epoch` with validation metric `metric`: (whether it is the best so far, whether the stage
        ends now)."""
        best = False
        if self.tracking(epoch):
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric, self.best_epoch, self.since, best = metric, epoch, 0, True
            else:
                self.since += 1
        return best, (self.tracking(epoch) and self.since >= self.patience) or epoch >= self.max_epochs

    def reason(self):
        return "patience" if self.since >= self.patience else "max_stage_epochs"

    def state(self):
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "since": self.since}

    def load(self, state):
        self.best_metric, self.best_epoch, self.since = state["best_metric"], state["best_epoch"], state["since"]


def snapshot(model):
    """A copy of every parameter and buffer (BatchNorm statistics included), on the CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def append_history(history, entry):
    """Append one epoch; a key new in this entry gets None for the earlier epochs, a key missing from it gets None,
    so every curve stays aligned with the epochs."""
    n_before = len(history.get("epoch", []))
    for k, v in entry.items():
        history.setdefault(k, [None] * n_before).append(v)
    for k, values in history.items():
        if len(values) == n_before:
            values.append(None)


# --------------------------------------------------------------------------------------------------- the run

def run(args):
    device = torch.device(args.device)
    name, settings = run_name(args), run_settings(args)
    json_path = os.path.join(args.results_dir, args.group, f"{name}.json")
    raw_dir = os.path.join(args.raw_dir, args.group, name)
    resume_path = os.path.join(args.ckpt_dir, args.group, f"{name}_resume.pt")

    if os.path.exists(json_path):
        with open(json_path) as f:
            done = json.load(f)
        if done["config"] != settings:
            raise SystemExit(f"{json_path} exists with different settings; delete it to run again")
        if done.get("frozen_bn") != FROZEN_BN:
            raise SystemExit(f"{json_path} was made before BatchNorm re-estimation at freezing (2026-11-22); "
                             f"move it away to run again")
        print(f"{name}: done already, skipped ({json_path})")
        return done
    for d in (os.path.dirname(json_path), raw_dir, os.path.dirname(resume_path)):
        os.makedirs(d, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # order and augmentation of the training images
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader, val_loader, test_loader = get_loaders(args.dataset, batch_size=args.batch_size, seed=0,
                                                        num_workers=args.workers, generator=generator)
    bn_loader = get_bn_loader(args.dataset, batch_size=args.batch_size, num_workers=args.workers)
    train_eval_loader = (get_train_eval_loader(args.dataset, args.train_eval_size, batch_size=args.batch_size,
                                               num_workers=args.workers) if args.train_eval_size > 0 else None)
    model, float_path = build_model(args, device)
    steps_per_epoch = len(train_loader) if args.max_batches is None else min(len(train_loader), args.max_batches)
    n_stages = len(network_units(model)[0]) - 1

    history, stages = {}, []
    steps = {"loss": [], "correct": [], "alpha": [], "stage": []}
    levels = (None, None)  # quantized weight levels two epochs ago and one epoch ago
    first_stage, resumed = 1, None
    failure = None

    def validate(stage, reest):
        """Validation with exact rounding everywhere (as deployed), with S in the optimized layer (`soft`), and
        after BatchNorm re-estimation if `reest`; the same on the fixed training images (train_eval_*)."""
        val = {k: None for k in ("val_acc", "val_loss", "val_acc_soft", "val_loss_soft", "val_acc_reest",
                                 "val_loss_reest")}
        val["val_acc"], val["val_loss"] = evaluate(model, val_loader, device, args.max_batches)
        if stage is not None:
            val["val_acc_soft"], val["val_loss_soft"] = evaluate(model, val_loader, device, args.max_batches, soft=True)
        copy_ = None
        if reest:
            copy_ = reestimated(model, bn_loader, device, args.max_batches)
            val["val_acc_reest"], val["val_loss_reest"] = evaluate(copy_, val_loader, device, args.max_batches)
        if train_eval_loader is not None:
            fit = {f"train_eval_{k}": None for k in ("acc", "loss", "acc_soft", "loss_soft", "acc_reest", "loss_reest")}
            fit["train_eval_acc"], fit["train_eval_loss"] = evaluate(model, train_eval_loader, device, args.max_batches)
            if stage is not None:
                fit["train_eval_acc_soft"], fit["train_eval_loss_soft"] = evaluate(
                    model, train_eval_loader, device, args.max_batches, soft=True)
            if copy_ is not None:
                fit["train_eval_acc_reest"], fit["train_eval_loss_reest"] = evaluate(
                    copy_, train_eval_loader, device, args.max_batches)
            val.update(fit)
        return val, copy_

    def level_entry():
        nonlocal levels
        now = weight_levels(model)
        changed, changed_layers, back, back_layers = level_changes(*levels, now)
        levels = (levels[1], now)
        return {"level_changed": changed, "level_back": back, "level_changed_layers": changed_layers,
                "level_back_layers": back_layers}

    def save_resume(k, epoch, stage=None, stopper=None, best=None):
        """The position after `epoch` of stage k (epoch 0: stage k has not started; k = n_stages + 1: all done)."""
        state = {"settings": settings, "frozen_bn": FROZEN_BN, "stage": k, "epoch": epoch, "model": model.state_dict(),
                 "optimizers": [o.state_dict() for o in stage.optimizers] if epoch else None,
                 "stopper": stopper.state() if epoch else None, "best": best if epoch else None,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         "generator": generator.get_state()},
                 "history": history, "stages": stages, "steps": steps, "levels": levels}
        tmp = resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)  # atomic: an interruption never leaves a broken file

    def n_epochs():
        return len(history["epoch"]) - 1  # without the start

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"{name}: {n_stages} stages of at most {args.max_stage_epochs} epochs x {steps_per_epoch} steps on "
          f"{device_name}", flush=True)

    if os.path.exists(resume_path):
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resumed["settings"] != settings:
            raise SystemExit(f"{resume_path} was made with different settings; delete it or use the same ones")
        if resumed.get("frozen_bn") != FROZEN_BN:
            raise SystemExit(f"{resume_path} was made before BatchNorm re-estimation at freezing (2026-11-22); "
                             f"delete it to start again")
        model.load_state_dict(resumed["model"])
        rng = resumed["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        generator.set_state(rng["generator"])
        history, stages, steps, levels = resumed["history"], resumed["stages"], resumed["steps"], tuple(resumed["levels"])
        first_stage = resumed["stage"]
        print(f"resuming at stage {first_stage}, after its epoch {resumed['epoch']}", flush=True)
    else:
        images, _ = next(iter(train_loader))  # EWGS: the first training batch, as in train_qat.py
        init_quantizers(model, images.to(device))
        t0 = time.perf_counter()
        val, reest = validate(None, reest=True)
        model.load_state_dict(reest.state_dict())  # the stem is frozen from stage 1 on: freeze it re-estimated
        append_history(history, {"epoch": 0, "stage": None, "stage_epoch": None, **val,
                                 **level_entry(), **quantizer_values(model), "train_seconds": 0.0,
                                 "eval_seconds": round(time.perf_counter() - t0, 2)})
        print(f"start: val {fmt(val['val_acc'])}, after re-estimation {fmt(val['val_acc_reest'])}", flush=True)

    for k in range(first_stage, n_stages + 1):
        stage = Stage(model, k, args, steps_per_epoch)
        stopper, best, first_epoch = EarlyStopping(args.stage_ramp_epochs, args.patience, args.max_stage_epochs), None, 1
        if resumed is not None and resumed["stage"] == k and resumed["epoch"]:
            for o, s in zip(stage.optimizers, resumed["optimizers"]):
                o.load_state_dict(s)
            stopper.load(resumed["stopper"])
            best, first_epoch = resumed["best"], resumed["epoch"] + 1
        resumed = None
        epoch = first_epoch - 1
        for epoch in range(first_epoch, args.max_stage_epochs + 1):
            t0 = time.perf_counter()
            step_loss, step_correct, step_size, step_alpha = train_epoch(model, stage, train_loader, epoch,
                                                                         steps_per_epoch, device, args.max_batches)
            steps["loss"] += step_loss
            steps["correct"] += step_correct
            steps["alpha"] += step_alpha
            steps["stage"] += [k] * len(step_loss)
            bad = [j for j, v in enumerate(step_loss) if not math.isfinite(v)]
            if bad:
                failure = f"non-finite loss in stage {k}, epoch {epoch}, step {bad[0]} of the epoch"
            train_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            # every epoch, ramp epochs included, is also validated after BatchNorm re-estimation (the supervisor,
            # 2026-11-21): the ramp epochs are never chosen, but their curves stay comparable with the others
            val, _ = validate(stage, reest=True)
            is_best, stop = stopper.update(epoch, val[args.stage_metric])  # ramp epochs: never best, no patience
            if is_best:
                best = snapshot(model)
            seen = sum(step_size)
            append_history(history, {
                "epoch": n_epochs() + 1, "stage": k, "stage_epoch": epoch,
                "tracking": stopper.tracking(epoch), "alpha": rounded(stage.alpha), **stage.lrs(),
                "train_loss": rounded(sum(v * s for v, s in zip(step_loss, step_size)) / seen),
                "train_acc": 100.0 * sum(step_correct) / seen, **val,
                "metric": val[args.stage_metric], "best": stopper.best_metric, "since_best": stopper.since,
                **level_entry(), **quantizer_values(model), "train_seconds": round(train_seconds, 2),
                "eval_seconds": round(time.perf_counter() - t0, 2)})

            seconds = [t + e for t, e in zip(history["train_seconds"][1:], history["eval_seconds"][1:])]
            typical = (sum(s["epochs"] for s in stages) / len(stages) if stages
                       else args.stage_ramp_epochs + args.patience)
            left = max(typical - epoch, 0) + typical * (n_stages - k)
            line = (f"stage {k:2d}/{n_stages} {stage.layer:15s} epoch {epoch:2d}  alpha {stage.alpha:5.2f}  "
                    f"lr {stage.lrs()['lr']:.1e}  train loss {fmt(history['train_loss'][-1], '.3f')} "
                    f"acc {fmt(history['train_acc'][-1])}")
            if train_eval_loader is not None:
                line += f"  train set {fmt(val['train_eval_acc'])} (S {fmt(val['train_eval_acc_soft'])})"
            line += f"  val {fmt(val['val_acc'])} (S {fmt(val['val_acc_soft'])})"
            if val["val_acc_reest"] is not None:
                line += f" (re-estimated {fmt(val['val_acc_reest'])})"
            line += ("  ramp" if not stopper.tracking(epoch) else
                     f"  best {fmt(stopper.best_metric)} @{stopper.best_epoch}  wait {stopper.since}/{args.patience}")
            line += f"  {seconds[-1]:.0f}s  ~{np.mean(seconds) * left / 60:.0f} min left"
            print(line, flush=True)
            if failure is not None:
                print(f"stopped: {failure}", flush=True)
                break
            if stop:
                break
            save_resume(k, epoch, stage, stopper, best)
            if args.stop_after is not None and n_epochs() >= args.stop_after:
                print(f"stopped after epoch {n_epochs()} (--stop-after); run again with the same arguments to resume")
                return None
        if failure is not None:
            break

        # the end of the stage: back to the best epoch, measure it, re-estimate every BatchNorm with exact rounding
        # over the full training split (layer k's statistics were collected with S), and freeze layer k with these
        # statistics (the next Stage freezes it)
        t0 = time.perf_counter()
        model.load_state_dict(best)
        levels = (None, weight_levels(model))
        after, reest = validate(None, reest=True)
        model.load_state_dict(reest.state_dict())
        in_stage = [i for i, s in enumerate(history["stage"]) if s == k]
        stages.append({"stage": k, "layer": stage.layer, "epochs": epoch, "best_epoch": stopper.best_epoch,
                       "best_metric": stopper.best_metric, "stopped_by": stopper.reason(),
                       "first_epoch": history["epoch"][in_stage[0]],
                       "val_acc": after["val_acc"], "val_loss": after["val_loss"],
                       "val_acc_reest": after["val_acc_reest"], "val_loss_reest": after["val_loss_reest"],
                       "train_eval_acc": after.get("train_eval_acc"), "train_eval_acc_reest": after.get("train_eval_acc_reest"),
                       "minutes": round((sum(history["train_seconds"][i] + history["eval_seconds"][i] for i in in_stage)
                                         + time.perf_counter() - t0) / 60, 2)})
        print(f"stage {k:2d} done: {epoch} epochs ({stopper.reason()}), back to epoch {stopper.best_epoch}: "
              f"val {fmt(after['val_acc'])}, after re-estimation {fmt(after['val_acc_reest'])}", flush=True)
        save_resume(k + 1, 0)
        if args.stop_after is not None and n_epochs() >= args.stop_after and k < n_stages:
            print(f"stopped after epoch {n_epochs()} (--stop-after); run again with the same arguments to resume")
            return None

    # the final evaluation: the deployed network (every layer rounds exactly), after the last stage's rollback.
    # The test set is used only here, and never for the development seed (PLAN.md, change 3).
    freeze_all(model)
    reest = reestimated(model, bn_loader, device, args.max_batches)
    keys = ["acc_reest", "acc", "loss_reest", "loss"]
    measured = {f"{split}_{k}": None for split in ("val", "test") for k in keys}
    measured["val_acc"], measured["val_loss"] = evaluate(model, val_loader, device, args.max_batches)
    measured["val_acc_reest"], measured["val_loss_reest"] = evaluate(reest, val_loader, device, args.max_batches)
    if args.seed != train_qat.DEV_SEED:
        measured["test_acc"], measured["test_loss"] = evaluate(model, test_loader, device, args.max_batches)
        measured["test_acc_reest"], measured["test_loss_reest"] = evaluate(reest, test_loader, device,
                                                                           args.max_batches)
    final = {key: measured[key] for k in keys for key in (f"val_{k}", f"test_{k}")}  # the order of train_qat.py
    start = {f"val_{k}": history[f"val_{k}"][0] for k in keys}
    if failure is None and final["val_acc_reest"] < start["val_acc_reest"]:
        failure = (f"ended below its start: validation accuracy after re-estimation "
                   f"{final['val_acc_reest']:.2f} < {start['val_acc_reest']:.2f}")

    bn_names = ("running_mean", "running_var", "num_batches_tracked")
    torch.save({"name": name, "config": settings,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "bn_reestimated": {k: v.detach().cpu() for k, v in reest.state_dict().items()
                                   if k.rsplit(".", 1)[-1] in bn_names}},
               os.path.join(raw_dir, "final.pt"))
    dtypes = {"correct": np.int32, "stage": np.int16}
    np.savez_compressed(os.path.join(raw_dir, "steps.npz"),
                        **{k: np.asarray(v, dtype=dtypes.get(k, np.float32)) for k, v in steps.items()})

    commit, dirty = git_state()
    result = {
        "name": name, "method": METHOD, "group": args.group, "frozen_bn": FROZEN_BN,
        "failed": failure is not None, "failure": failure,
        "final": final, "start": start, "config": settings, "n_epochs": n_epochs(), "stages": stages,
        "layers": [n for n, _ in quantized_convs(model)], "steps_per_epoch": steps_per_epoch,
        "float_model": {"file": os.path.relpath(float_path, HERE), "sha256": file_hash(float_path)},
        "val_split": split_fingerprint(val_loader.dataset.indices),
        "git_commit": commit, "git_dirty": dirty, "device": device_name,
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "train_minutes": round(sum(history["train_seconds"]) / 60, 2),
        "total_minutes": round((sum(history["train_seconds"]) + sum(history["eval_seconds"])) / 60, 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "history": history,
    }
    write_json(result, json_path)  # written last: its existence marks the run as done
    if os.path.exists(resume_path):
        os.remove(resume_path)
    summary = f"val {final['val_acc_reest']:.2f} after re-estimation (without: {final['val_acc']:.2f})"
    if final["test_acc_reest"] is not None:
        summary += f", test {final['test_acc_reest']:.2f} (without: {final['test_acc']:.2f})"
    else:
        summary += " (development seed: test set not evaluated)"
    print(f"done after {n_epochs()} epochs: " + summary + (f"; FAILED: {failure}" if failure else "")
          + f"\n  {json_path}", flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
