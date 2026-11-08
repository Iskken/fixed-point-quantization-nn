"""
Train the full-precision ResNet-20 once per seed; every QAT method starts from this model.

Recipe of the official EWGS code (CIFAR10/train_fp.py): SGD, momentum 0.9, lr 0.1 with cosine decay
to 0, weight decay 1e-4, 400 epochs, batch size 256 (on our 45k training split). QAT starts from the
model after the last epoch (no selection on validation or test).

    python -m piecewise_nonlinearity.train_float --seed 0

The run can be interrupted at any time: it saves a resume checkpoint every --save-every epochs and
continues from it when started again with the same arguments. A resumed run is not bit-identical to an
uninterrupted one (data order and cuDNN), which is fine for a float starting point.

Outputs:
  <pretrained-dir>/resnet20_<dataset>_seed<N>.pt   final weights + metadata (small, meant to be shared)
  <ckpt-dir>/resnet20_<dataset>_seed<N>_resume.pt  model + optimizer + scheduler state (local only)
  results/float/resnet20_<dataset>_seed<N>.json    config, per-epoch curves, final val/test accuracy
"""

import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from piecewise_nonlinearity.datasets import DATASETS, download, get_loaders
from piecewise_nonlinearity.models.resnet20 import ResNet20

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total, loss_sum / total


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10, help="epochs between resume checkpoints")
    p.add_argument("--ckpt-dir", default=os.path.join(HERE, "checkpoints"))
    p.add_argument("--pretrained-dir", default=os.path.join(HERE, "pretrained"))
    p.add_argument("--results-dir", default=os.path.join(HERE, "results", "float"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    for d in (args.ckpt_dir, args.pretrained_dir, args.results_dir):
        os.makedirs(d, exist_ok=True)
    name = f"resnet20_{args.dataset}_seed{args.seed}"
    resume_path = os.path.join(args.ckpt_dir, f"{name}_resume.pt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    download(args.dataset)
    # the train/validation split is the same for every training seed (split seed 0)
    train_loader, val_loader, test_loader = get_loaders(args.dataset, batch_size=args.batch_size,
                                                        seed=0, num_workers=args.workers)

    model = ResNet20(num_classes=DATASETS[args.dataset]["n_classes"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    history = {"epoch": [], "lr": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": [],
               "seconds": []}
    start_epoch = 0
    # the settings that define the run (paths and worker count may differ between machines)
    run_args = {k: v for k, v in vars(args).items() if k not in ("ckpt_dir", "pretrained_dir",
                                                                 "results_dir", "workers")}

    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["args"] != run_args:
            raise SystemExit(f"{resume_path} was made with different settings; delete it or use the same ones.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"].cpu())  # map_location moved it to the GPU
        history, start_epoch = state["history"], state["epoch"]
        print(f"resuming {name} from epoch {start_epoch}")

    def save_resume(epoch):
        tmp = resume_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng_cpu": torch.get_rng_state(), "history": history, "epoch": epoch, "args": run_args}, tmp)
        os.replace(tmp, resume_path)  # atomic, so an interruption never leaves a broken file

    print(f"{name}: {args.epochs} epochs on {device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        lr = opt.param_groups[0]["lr"]
        sched.step()
        val_acc, val_loss = evaluate(model, val_loader, device)
        sec = time.perf_counter() - t0
        for k, v in zip(history, (epoch + 1, lr, loss_sum / total, 100.0 * correct / total, val_acc, val_loss, sec)):
            history[k].append(v)
        eta = sec * (args.epochs - epoch - 1) / 60
        print(f"epoch {epoch + 1:3d}/{args.epochs}  lr {lr:.4f}  train loss {loss_sum / total:.3f} "
              f"acc {100.0 * correct / total:5.2f}  val acc {val_acc:5.2f}  {sec:4.1f}s  ~{eta:.0f} min left",
              flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_resume(epoch + 1)

    test_acc, test_loss = evaluate(model, test_loader, device)
    meta = {"name": name, "config": run_args, "final_val_acc": history["val_acc"][-1], "test_acc": test_acc,
            "test_loss": test_loss, "git_commit": git_commit(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "train_minutes": sum(history["seconds"]) / 60}
    torch.save({"model": model.state_dict(), "meta": meta},
               os.path.join(args.pretrained_dir, f"{name}.pt"))
    with open(os.path.join(args.results_dir, f"{name}.json"), "w") as f:
        json.dump({**meta, "history": history}, f, indent=1)
    print(f"done: final val acc {meta['final_val_acc']:.2f}, test acc {test_acc:.2f}; "
          f"weights in {os.path.join(args.pretrained_dir, name + '.pt')}")


if __name__ == "__main__":
    main()
