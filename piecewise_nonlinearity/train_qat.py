"""
QAT fine-tuning from the float ResNet-20: one run per call, the same protocol for every method.

    python -m piecewise_nonlinearity.train_qat --group tune --method piecewise --bits 2/2 --seed 0 \
        --epochs 60 --lr 1e-3 --alpha-end 2 --beta 2
    python -m piecewise_nonlinearity.train_qat --group tune --method ste --bits 2/2 --seed 0 --epochs 60 --lr 1e-3

Methods: ste, and piecewise (our staircase); ewgs and dsq are added in step 8. --bits W/A, with 32 for
float: 2/32 quantizes only the weights, 32/2 only the activations.

One run (PLAN.md, changes 1-5):
  1. ResNet-20 with quantized block convolutions, loaded from the float model of the run's seed
     (pretrained/resnet20_<dataset>_seed<N>.pt).
  2. The quantizers are initialized as in EWGS, on the first augmented training batch, with exact
     rounding: every method with the same seed starts from the same quantized network ("the start",
     logged as epoch 0).
  3. The EWGS training recipe: Adam for the network (--lr, --weight-decay) and Adam for l, u and the output
     scales (--lr-q, no weight decay), both with cosine decay to 0, stepped once per epoch.
     Staircase: alpha rises linearly from --alpha-start to --alpha-end over the first --ramp-fraction of
     all steps, updated before every step; beta stays fixed.
  4. After the last epoch, with no selection: validation and test accuracy with exact rounding, without and
     with BatchNorm re-estimation (models/quant_layers.reestimate_bn, on a copy). Validation after
     re-estimation is the tuning metric, test after re-estimation the primary metric. The test set is
     used only here.
A run that ends below its start (validation after re-estimation) or hits a non-finite loss is marked failed.

Logged per epoch: training loss and accuracy (with the training forward), validation accuracy with exact
rounding, for the staircase also with its forward S (the forward gap), with re-estimation every
--reestimate-every epochs, the share of weights that changed level and the share that went back to the
level of two epochs before, l, u and the output scale of every layer, alpha, and the time.

The order and the augmentation of the training images depend only on --seed (the training loader has its
own generator, and with --workers > 0 the augmentation runs in the workers), so they are the same for every
method and do not change with how often we evaluate. cuDNN is not deterministic, so a rerun on the GPU can
differ slightly.

Outputs; <name> contains every setting, e.g. piecewise_w2a2_lr1e-3_lrq1e-5_wd1e-4_a2b2-exact-r0.5_e60_s0:
  results/qat/<group>/<name>.json            config, machine, start, per-epoch curves, final metrics (committed)
  results/raw/qat/<group>/<name>/final.pt    final weights, and the re-estimated BatchNorm statistics (local)
  results/raw/qat/<group>/<name>/steps.npz   loss, correct predictions and alpha at every step (local)
  checkpoints/qat/<group>/<name>_resume.pt   resume checkpoint, deleted when the run finishes (local)

A run whose JSON already exists with the same settings is skipped, so a queue of runs can be restarted.
An interrupted run continues from its last resume checkpoint when started again with the same arguments;
on the GPU it is not bit-identical to an uninterrupted run.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from piecewise_nonlinearity.datasets import DATASETS, get_bn_loader, get_loaders, split_fingerprint
from piecewise_nonlinearity.models.quant_layers import (QConv2d, init_quantizers, load_float_weights,
                                                         qconv_factory, quantizers, reestimate_bn,
                                                         split_parameters)
from piecewise_nonlinearity.models.resnet20 import ResNet20
from piecewise_nonlinearity.quantizers.piecewise import BACKWARDS, alpha_at, set_staircase

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("ste", "piecewise", "ewgs", "dsq")
IMPLEMENTED = ("ste", "piecewise")  # EWGS and DSQ come with step 8
GROUPS = ("pilot", "tune", "main", "ablation", "repro", "smoke")
# the staircase's settings and their defaults; other methods reject them, so no setting is silently ignored
STAIRCASE = {"alpha_end": 2.0, "beta": 2.0, "backward": "exact", "alpha_start": 1.0, "ramp_fraction": 0.5}
# arguments that do not change what is trained: paths, machine, checkpointing, tests
NOT_SETTINGS = ("results_dir", "raw_dir", "ckpt_dir", "pretrained_dir", "workers", "device", "save_every",
                "stop_after")


# ----------------------------------------------------------------------------------------------- arguments

def parse_bits(text):
    """'W/A' -> (w_bits, a_bits), with None for 32 (that side stays in float)."""
    try:
        w, a = (int(t) for t in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--bits must look like 2/2 or 4/32, got {text!r}") from None
    if not all(1 <= b <= 8 or b == 32 for b in (w, a)):
        raise argparse.ArgumentTypeError(f"bits must be 1 to 8, or 32 for float; got {text!r}")
    if w == 32 and a == 32:
        raise argparse.ArgumentTypeError("--bits 32/32 quantizes nothing; that is train_float.py")
    return (None if w == 32 else w), (None if a == 32 else a)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--bits", required=True, help="W/A, e.g. 2/2 or 4/4; 32 = float, so 2/32 is weights only")
    p.add_argument("--seed", type=int, default=0, help="QAT seed, and the float model the run starts from")
    p.add_argument("--group", required=True, choices=GROUPS, help="subfolder of the results")
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the network (Adam)")
    p.add_argument("--lr-q", type=float, default=1e-5, help="learning rate of l, u and the output scales (Adam)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="on the network parameters only")
    p.add_argument("--batch-size", type=int, default=256)
    s = p.add_argument_group("staircase, --method piecewise only")
    s.add_argument("--alpha-end", type=float, help="alpha after the ramp (default 2)")
    s.add_argument("--beta", type=float, help="fixed for the whole run (default 2)")
    s.add_argument("--backward", choices=BACKWARDS,
                   help="exact: the slopes of S (our method); ste: slope 1 (ablation). Default exact")
    s.add_argument("--alpha-start", type=float, help="alpha at the first step (default 1)")
    s.add_argument("--ramp-fraction", type=float, help="share of all steps over which alpha rises (default 0.5)")
    p.add_argument("--reestimate-every", type=int, default=10,
                   help="epochs between validation points with BatchNorm re-estimation (0: start and end only)")
    p.add_argument("--save-every", type=int, default=10, help="epochs between resume checkpoints (0: never)")
    p.add_argument("--max-batches", type=int, help="smoke tests: cut every epoch and evaluation to this many batches")
    p.add_argument("--stop-after", type=int, help="tests: stop after this epoch, as if interrupted")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default=os.path.join(HERE, "results", "qat"))
    p.add_argument("--raw-dir", default=os.path.join(HERE, "results", "raw", "qat"))
    p.add_argument("--ckpt-dir", default=os.path.join(HERE, "checkpoints", "qat"))
    p.add_argument("--pretrained-dir", default=os.path.join(HERE, "pretrained"))
    args = p.parse_args(argv)

    if args.method not in IMPLEMENTED:
        p.error(f"--method {args.method} is added in step 8 (quantizers/{args.method}.py); not implemented yet")
    try:
        args.w_bits, args.a_bits = parse_bits(args.bits)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    given = [k for k in STAIRCASE if getattr(args, k) is not None]
    if args.method == "piecewise":
        for k, default in STAIRCASE.items():
            if getattr(args, k) is None:
                setattr(args, k, default)
        if not 0.0 < args.ramp_fraction <= 1.0:
            p.error("--ramp-fraction must be in (0, 1]")
    elif given:
        p.error(f"{', '.join('--' + k.replace('_', '-') for k in given)} only apply to --method piecewise")
    return args


def short(x):
    """A number for run names: 1e-3 -> '1e-3', 3e-4 -> '3e-4', 0.5 -> '0.5', 2.0 -> '2', 0 -> '0'."""
    if x == 0:
        return "0"
    if 0.1 <= abs(x) < 1000:
        return f"{x:g}"
    mantissa, exponent = f"{x:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def run_name(args):
    """Every setting of the run, so two different runs never share a name."""
    w, a = args.bits.split("/")
    parts = [args.method, f"w{w}a{a}", f"lr{short(args.lr)}", f"lrq{short(args.lr_q)}",
             f"wd{short(args.weight_decay)}"]
    if args.method == "piecewise":
        stair = f"a{short(args.alpha_end)}b{short(args.beta)}-{args.backward}-r{short(args.ramp_fraction)}"
        if args.alpha_start != STAIRCASE["alpha_start"]:
            stair += f"-from{short(args.alpha_start)}"
        parts.append(stair)
    parts += [f"e{args.epochs}", f"s{args.seed}"]
    if args.batch_size != 256:
        parts.append(f"bs{args.batch_size}")
    if args.max_batches is not None:
        parts.append(f"mb{args.max_batches}")
    name = "_".join(parts)
    return name if args.dataset == "cifar10" else f"{args.dataset}_{name}"


# ------------------------------------------------------------------------------------------ model and data

def build_model(args, device):
    """The quantized ResNet-20 with the float weights of the run's seed. Returns (model, float model path)."""
    path = os.path.join(args.pretrained_dir, f"resnet20_{args.dataset}_seed{args.seed}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; train it first with "
                         f"python -m piecewise_nonlinearity.train_float --seed {args.seed}")
    stair = ({"alpha": args.alpha_start, "beta": args.beta, "backward": args.backward}
             if args.method == "piecewise" else {})
    model = ResNet20(num_classes=DATASETS[args.dataset]["n_classes"],
                     conv_layer=qconv_factory(args.method, args.w_bits, args.a_bits, **stair))
    load_float_weights(model, torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.to(device), path


def quantized_convs(model):
    """(name, module) of every quantized convolution, in model order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, QConv2d) and (m.wq is not None or m.aq is not None)]


def batches(loader, max_batches=None):
    return loader if max_batches is None else itertools.islice(loader, max_batches)


# ------------------------------------------------------------------------------------------------ measuring

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, soft=False):
    """
    (accuracy in %, mean cross-entropy) with BatchNorm in eval mode. The quantizers round exactly, or with
    soft=True keep their training forward (for the staircase: S), which measures the forward gap.
    """
    model.eval()
    if soft:
        for q in quantizers(model):
            q.train()
    correct = total = 0
    loss_sum = 0.0
    for x, y in batches(loader, max_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.eval()
    return 100.0 * correct / total, rounded(loss_sum / total)


def reestimated(model, bn_loader, device, max_batches=None):
    """A copy of the model whose BatchNorm statistics are re-estimated with exact rounding (PLAN.md, change 5)."""
    m = copy.deepcopy(model)
    reestimate_bn(m, batches(bn_loader, max_batches), device=device)
    return m


@torch.no_grad()
def weight_levels(model):
    """The level of every quantized weight under exact rounding: one int16 tensor per layer, on the CPU."""
    levels = []
    for _, m in quantized_convs(model):
        if m.wq is not None:
            q = m.wq
            x_n = ((m.weight - q.l) / (q.u - q.l)).clamp(0.0, 1.0)
            levels.append(torch.round(x_n * (q.levels - 1)).to(torch.int16).cpu())
    return levels


def level_changes(two_ago, one_ago, now):
    """
    (share of quantized weights whose level changed since the last epoch, the same per layer,
     share that went back to the level of two epochs before, the same per layer); None where there is
    nothing to compare. "Went back" means: it left that level in the previous epoch and returned in this one.
    """
    if not now or one_ago is None:
        return None, None, None, None
    sizes = [n.numel() for n in now]
    changed = [(n != b).sum().item() for n, b in zip(now, one_ago)]
    result = [rounded(sum(changed) / sum(sizes)), [rounded(c / s) for c, s in zip(changed, sizes)]]
    if two_ago is None:
        return (*result, None, None)
    back = [((n == b2) & (b != b2)).sum().item() for n, b, b2 in zip(now, one_ago, two_ago)]
    return (*result, rounded(sum(back) / sum(sizes)), [rounded(c / s) for c, s in zip(back, sizes)])


@torch.no_grad()
def quantizer_values(model):
    """l and u of every weight and activation quantizer and every output scale, as lists over the layers."""
    values = {"w_l": [], "w_u": [], "a_l": [], "a_u": [], "output_scale": []}
    for _, m in quantized_convs(model):
        for side, q in (("w", m.wq), ("a", m.aq)):
            values[f"{side}_l"].append(rounded(q.l) if q is not None else None)
            values[f"{side}_u"].append(rounded(q.u) if q is not None else None)
        values["output_scale"].append(rounded(m.output_scale.abs()))  # the forward uses |output_scale|
    return values


def rounded(x, digits=6):
    """A plain float with `digits` significant digits, or None if it is not finite (keeps the JSON readable)."""
    x = float(x)
    return float(f"{x:.{digits}g}") if math.isfinite(x) else None


# ---------------------------------------------------------------------------------------------------- hooks

class MethodHooks:
    """What a method does during training besides the optimizer steps (the STE: nothing). EWGS (Hessian
    update of delta every 10 epochs, in on_epoch_start) and DSQ add their own subclass in step 8."""

    def on_epoch_start(self, model, epoch):
        pass

    def on_step(self, model, step):
        pass

    def logged(self):
        """Values recorded once per epoch in the history, and once per step in steps.npz."""
        return {}


class StaircaseHooks(MethodHooks):
    """alpha rises linearly from alpha_start to alpha_end over the first ramp_fraction of all training steps,
    set before every step; beta stays fixed (PLAN.md, change 2)."""

    def __init__(self, args, total_steps):
        self.start, self.end, self.ramp = args.alpha_start, args.alpha_end, args.ramp_fraction
        self.total_steps = total_steps
        self.alpha = self.start

    def on_step(self, model, step):
        self.alpha = alpha_at(step / self.total_steps, self.start, self.end, "linear", self.ramp)
        set_staircase(model, alpha=self.alpha)

    def logged(self):
        return {"alpha": self.alpha}


# ---------------------------------------------------------------------------------------------------- saving

def git_state():
    """(short commit, whether any .py file in piecewise_nonlinearity/ differs from it)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "*.py"], cwd=HERE,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return commit, bool(dirty)
    except Exception:
        return "unknown", None


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def write_json(obj, path):
    """Indented JSON with every list of plain values on one line, written atomically."""
    text = json.dumps(obj, indent=1)
    text = re.sub(r"\[([^\[\]{}\"]*)\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text + "\n")
    os.replace(tmp, path)


def fmt(x, spec="5.2f"):
    return "-" if x is None else format(x, spec)


# --------------------------------------------------------------------------------------------------- the run

def run(args):
    device = torch.device(args.device)
    name = run_name(args)
    settings = {k: v for k, v in vars(args).items() if k not in NOT_SETTINGS}
    json_path = os.path.join(args.results_dir, args.group, f"{name}.json")
    raw_dir = os.path.join(args.raw_dir, args.group, name)
    resume_path = os.path.join(args.ckpt_dir, args.group, f"{name}_resume.pt")

    if os.path.exists(json_path):
        with open(json_path) as f:
            done = json.load(f)
        if done["config"] != settings:
            raise SystemExit(f"{json_path} exists with different settings; delete it to run again")
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
    model, float_path = build_model(args, device)
    steps_per_epoch = len(train_loader) if args.max_batches is None else min(len(train_loader), args.max_batches)
    total_steps = args.epochs * steps_per_epoch
    hooks = StaircaseHooks(args, total_steps) if args.method == "piecewise" else MethodHooks()
    soft = args.method == "piecewise"  # its training forward (S) differs from exact rounding

    net_params, quant_params = split_parameters(model)
    opt_net = torch.optim.Adam(net_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_q = torch.optim.Adam(quant_params, lr=args.lr_q)
    optimizers = (opt_net, opt_q)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs, eta_min=0.0) for o in optimizers]

    history, steps = {}, {"loss": [], "correct": []}
    levels = (None, None)  # quantized weight levels two epochs ago and one epoch ago
    start_epoch, failure, reest = 0, None, None

    def validate(epoch, last=False):
        """Validation metrics; with BatchNorm re-estimation at the start, every --reestimate-every epochs and
        at the end. Returns (metrics, the re-estimated copy or None)."""
        val = {"val_acc": None, "val_loss": None, "val_acc_soft": None, "val_loss_soft": None,
               "val_acc_reest": None, "val_loss_reest": None}
        val["val_acc"], val["val_loss"] = evaluate(model, val_loader, device, args.max_batches)
        if soft:
            val["val_acc_soft"], val["val_loss_soft"] = evaluate(model, val_loader, device, args.max_batches, soft=True)
        copy_ = None
        if last or epoch == 0 or (args.reestimate_every > 0 and epoch % args.reestimate_every == 0):
            copy_ = reestimated(model, bn_loader, device, args.max_batches)
            val["val_acc_reest"], val["val_loss_reest"] = evaluate(copy_, val_loader, device, args.max_batches)
        return val, copy_

    def log_epoch(epoch, train_loss, train_acc, lrs, val, train_seconds, eval_seconds):
        nonlocal levels
        now = weight_levels(model)
        changed, changed_layers, back, back_layers = level_changes(*levels, now)
        levels = (levels[1], now)
        entry = {"epoch": epoch, "lr": lrs[0], "lr_q": lrs[1], "train_loss": train_loss, "train_acc": train_acc,
                 **val, "level_changed": changed, "level_back": back, "level_changed_layers": changed_layers,
                 "level_back_layers": back_layers, **quantizer_values(model), **hooks.logged(),
                 "train_seconds": round(train_seconds, 2), "eval_seconds": round(eval_seconds, 2)}
        for k, v in entry.items():
            history.setdefault(k, []).append(v)
        return changed

    def save_resume(epoch):
        state = {"settings": settings, "epoch": epoch, "model": model.state_dict(),
                 "optimizers": [o.state_dict() for o in optimizers],
                 "schedulers": [s.state_dict() for s in schedulers],
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         "generator": generator.get_state()},
                 "history": history, "steps": steps, "levels": levels}
        tmp = resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)  # atomic: an interruption never leaves a broken file

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"{name}: {args.epochs} epochs x {steps_per_epoch} steps on {device_name}", flush=True)

    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise SystemExit(f"{resume_path} was made with different settings; delete it or use the same ones")
        model.load_state_dict(state["model"])
        for o, s in zip(optimizers, state["optimizers"]):
            o.load_state_dict(s)
        for sch, s in zip(schedulers, state["schedulers"]):
            sch.load_state_dict(s)
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        generator.set_state(rng["generator"])
        history, steps, levels, start_epoch = state["history"], state["steps"], tuple(state["levels"]), state["epoch"]
        print(f"resuming after epoch {start_epoch}", flush=True)
    else:
        images, _ = next(iter(train_loader))  # EWGS: the first training batch
        init_quantizers(model, images.to(device))
        t0 = time.perf_counter()
        val, _ = validate(0)
        log_epoch(0, None, None, (None, None), val, 0.0, time.perf_counter() - t0)
        print(f"start: val {fmt(val['val_acc'])}, after re-estimation {fmt(val['val_acc_reest'])}"
              + (f", with S {fmt(val['val_acc_soft'])}" if soft else ""), flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        hooks.on_epoch_start(model, epoch)
        model.train()
        t0 = time.perf_counter()
        step_loss, step_correct, step_size = [], [], []  # kept on the GPU: no synchronization inside the epoch
        for i, (x, y) in enumerate(batches(train_loader, args.max_batches)):
            hooks.on_step(model, (epoch - 1) * steps_per_epoch + i)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            for o in optimizers:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in optimizers:
                o.step()
            step_loss.append(loss.detach())
            step_correct.append((out.argmax(1) == y).sum())
            step_size.append(y.numel())
            for k, v in hooks.logged().items():
                steps.setdefault(k, []).append(v)
        step_loss = torch.stack(step_loss).tolist()
        step_correct = torch.stack(step_correct).tolist()
        steps["loss"] += step_loss
        steps["correct"] += step_correct
        bad = [j for j, v in enumerate(step_loss) if not math.isfinite(v)]
        if bad:
            failure = f"non-finite loss in epoch {epoch}, step {(epoch - 1) * steps_per_epoch + bad[0]}"
        seen = sum(step_size)
        loss_sum = sum(v * n for v, n in zip(step_loss, step_size))
        correct = sum(step_correct)
        lrs = [rounded(o.param_groups[0]["lr"]) for o in optimizers]
        for sch in schedulers:
            sch.step()
        train_seconds = time.perf_counter() - t0

        last = epoch == args.epochs or failure is not None
        t0 = time.perf_counter()
        val, reest = validate(epoch, last)
        changed = log_epoch(epoch, rounded(loss_sum / seen) if seen else None, 100.0 * correct / seen if seen else None,
                            lrs, val, train_seconds, time.perf_counter() - t0)
        seconds = [t + e for t, e in zip(history["train_seconds"][1:], history["eval_seconds"][1:])]
        line = f"epoch {epoch:3d}/{args.epochs}  lr {lrs[0]:.2e}"
        if "alpha" in history:
            line += f"  alpha {history['alpha'][-1]:.3f}"
        line += f"  train loss {fmt(history['train_loss'][-1], '.3f')} acc {fmt(history['train_acc'][-1])}"
        line += f"  val {fmt(val['val_acc'])}"
        if soft:
            line += f" (S {fmt(val['val_acc_soft'])})"
        if val["val_acc_reest"] is not None:
            line += f" (re-estimated {fmt(val['val_acc_reest'])})"
        if changed is not None:
            line += f"  levels changed {changed:.2%}"
        line += f"  {seconds[-1]:.0f}s  ~{np.mean(seconds) * (args.epochs - epoch) / 60:.0f} min left"
        print(line, flush=True)
        if failure is not None:
            print(f"stopped: {failure}", flush=True)
        if last:
            break
        if args.stop_after == epoch:
            save_resume(epoch)
            print(f"stopped after epoch {epoch} (--stop-after); run again with the same arguments to resume")
            return None
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_resume(epoch)

    # the final evaluation: the model after the last epoch, no selection; the test set is used only here
    if reest is None:
        reest = reestimated(model, bn_loader, device, args.max_batches)
    test = {}
    test["test_acc"], test["test_loss"] = evaluate(model, test_loader, device, args.max_batches)
    test["test_acc_reest"], test["test_loss_reest"] = evaluate(reest, test_loader, device, args.max_batches)
    if soft:
        test["test_acc_soft"], test["test_loss_soft"] = evaluate(model, test_loader, device, args.max_batches, soft=True)
    last_val = {k: history[k][-1] for k in history if k.startswith("val_")}
    keys = ["acc_reest", "acc", "loss_reest", "loss"] + (["acc_soft", "loss_soft"] if soft else [])
    final = {}
    for k in keys:  # validation after re-estimation: the tuning metric; test after re-estimation: the primary one
        final[f"val_{k}"], final[f"test_{k}"] = last_val[f"val_{k}"], test[f"test_{k}"]
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
    np.savez_compressed(os.path.join(raw_dir, "steps.npz"),
                        **{k: np.asarray(v, dtype=np.int32 if k == "correct" else np.float32) for k, v in steps.items()})

    commit, dirty = git_state()
    result = {
        "name": name, "group": args.group, "failed": failure is not None, "failure": failure,
        "final": final, "start": start, "config": settings,
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
    print(f"done: val {final['val_acc_reest']:.2f} and test {final['test_acc_reest']:.2f} after re-estimation "
          f"(test without: {final['test_acc']:.2f})" + (f"; FAILED: {failure}" if failure else "")
          + f"\n  {json_path}", flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
