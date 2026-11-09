"""
Check that this machine matches the reference setup, so runs on both laptops are comparable.

    python -m piecewise_nonlinearity.check_env           # versions, code, data, split, float models (~15 s)
    python -m piecewise_nonlinearity.check_env --bench   # also time QAT training on this GPU (~2 min)

Every check prints OK, MISMATCH or INFO. Fix every MISMATCH before running experiments that will be
compared across machines. The expected values come from the first laptop (2026-11-09).
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time

import numpy as np
import PIL
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))

EXPECTED = {
    "python": "3.11",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "numpy": "2.4.2",
    "pillow": "12.1.1",
    "cuda": "12.8",
    "cudnn": 91002,
    # sha256 prefix of the validation indices (datasets.split_fingerprint)
    "val_split": "bf1f82ed5656c06b",
    # sha256 prefixes of the committed float models
    "pretrained": {0: "0a01b980fce6dcdc", 1: "8c87e9ca4587165a", 2: "ad84bf48bdb59200"},
    # validation accuracy of the seed-0 float model (results/float/resnet20_cifar10_seed0.json)
    "float_seed0_val_acc": 91.66,
    # the same after BatchNorm re-estimation (models/quant_layers.reestimate_bn + datasets.get_bn_loader)
    "float_seed0_val_acc_reestimated": 91.68,
}

failures = []


def report(name, ok, detail):
    status = "OK      " if ok is True else ("INFO    " if ok is None else "MISMATCH")
    if ok is False:
        failures.append(name)
    print(f"{status} {name:<22} {detail}")


def base(version):
    return version.split("+")[0]  # "2.10.0+cu128" -> "2.10.0"


def check_versions():
    py = platform.python_version()
    report("python", py.startswith(EXPECTED["python"] + "."), py)
    for name, mod in (("torch", torch), ("torchvision", torchvision), ("numpy", np), ("pillow", PIL)):
        report(name, base(mod.__version__) == EXPECTED[name], mod.__version__)
    report("cuda (torch build)", torch.version.cuda == EXPECTED["cuda"], torch.version.cuda)
    cudnn = torch.backends.cudnn.version()
    report("cudnn", cudnn == EXPECTED["cudnn"], cudnn)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report("gpu", None, f"{p.name}, {p.total_memory / 2**30:.1f} GB")
    else:
        report("gpu", False, "CUDA not available")


def check_code():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=HERE).decode().strip()
        report("git commit", None, commit + (" (uncommitted changes in piecewise_nonlinearity/)" if dirty else ""))
    except Exception as e:
        report("git commit", False, f"git not available: {e}")


def check_data_and_split():
    from piecewise_nonlinearity.datasets import DATA_DIR, DATASETS, split_fingerprint, split_indices
    for name, spec in DATASETS.items():
        try:
            ok = all(spec["cls"](DATA_DIR, train=t, download=False)._check_integrity() for t in (True, False))
            report(f"data {name}", ok, "MD5 checksums match" if ok else "checksum failed: re-download")
        except RuntimeError:
            report(f"data {name}", False, "not downloaded: run python -m piecewise_nonlinearity.datasets")
    fp = split_fingerprint(split_indices(50000)[1])
    report("validation split", fp == EXPECTED["val_split"], fp)


def check_float_models():
    from piecewise_nonlinearity.datasets import get_loaders
    from piecewise_nonlinearity.models.resnet20 import ResNet20
    for seed, expected in EXPECTED["pretrained"].items():
        path = os.path.join(HERE, "pretrained", f"resnet20_cifar10_seed{seed}.pt")
        if not os.path.exists(path):
            report(f"float model seed {seed}", False, "missing: git pull")
            continue
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        report(f"float model seed {seed}", digest == expected, digest)
    # end to end: the seed-0 float model on the validation split must give the recorded accuracy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet20()
    ckpt = torch.load(os.path.join(HERE, "pretrained", "resnet20_cifar10_seed0.pt"), map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    _, val_loader, _ = get_loaders("cifar10", num_workers=2)

    def val_acc(m):
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += (m(x.to(device)).argmax(1).cpu() == y).sum().item()
                total += y.numel()
        return 100.0 * correct / total, total

    acc, total = val_acc(model)
    diff_images = round(abs(acc - EXPECTED["float_seed0_val_acc"]) * total / 100)
    report("float seed 0 val acc", diff_images <= 2,
           f"{acc:.2f} (expected {EXPECTED['float_seed0_val_acc']:.2f}; {diff_images} images differ)")

    # BatchNorm re-estimation: on the float model it must leave the accuracy (almost) unchanged. A large drop
    # means the re-estimation data is wrong (unaugmented training images gave 87.74).
    import copy

    from piecewise_nonlinearity.datasets import get_bn_loader
    from piecewise_nonlinearity.models.quant_layers import reestimate_bn
    reestimated = copy.deepcopy(model)
    reestimate_bn(reestimated, get_bn_loader("cifar10", num_workers=2), device=device)
    acc_re, _ = val_acc(reestimated)
    expected_re = EXPECTED["float_seed0_val_acc_reestimated"]
    diff_re = round(abs(acc_re - expected_re) * total / 100)
    report("BN re-estimation", abs(acc_re - acc) <= 0.3 and diff_re <= 2,
           f"float seed 0 val acc {acc_re:.2f} after re-estimation (expected {expected_re:.2f}, "
           f"and within 0.3 of {acc:.2f})")


def bench():
    """Seconds per training epoch (176 batches of 256) for STE and the staircase at 4/4."""
    import torch.nn.functional as F
    from piecewise_nonlinearity.models.quant_layers import (init_quantizers, load_float_weights, qconv_factory,
                                                            split_parameters)
    from piecewise_nonlinearity.models.resnet20 import ResNet20
    torch.backends.cudnn.benchmark = True
    state = torch.load(os.path.join(HERE, "pretrained", "resnet20_cifar10_seed0.pt"), map_location="cpu",
                       weights_only=True)["model"]
    x = torch.randn(256, 3, 32, 32, device="cuda")
    y = torch.randint(0, 10, (256,), device="cuda")
    for name, method, kw in (("STE 4/4", "ste", {}), ("staircase 4/4", "piecewise", {"alpha": 2.0, "beta": 2.0})):
        model = ResNet20(conv_layer=qconv_factory(method, 4, 4, **kw))
        load_float_weights(model, state)
        model = model.cuda()
        init_quantizers(model, x)
        net, quant = split_parameters(model)
        opt = torch.optim.Adam([{"params": net, "lr": 1e-3}, {"params": quant, "lr": 1e-5}])
        model.train()
        for i in range(60):
            if i == 10:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        per_batch = (time.perf_counter() - t0) / 50
        report(f"bench {name}", None, f"{per_batch * 1e3:.0f} ms per batch, ~{per_batch * 176:.0f} s per epoch "
                                      "(reference laptop, same benchmark: STE ~15 s, staircase ~22 s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", action="store_true", help="also time QAT training on this GPU (~2 min)")
    args = p.parse_args()
    print(f"checking {platform.node()} ({sys.executable})")
    check_versions()
    check_code()
    check_data_and_split()
    check_float_models()
    if args.bench:
        if torch.cuda.is_available():
            bench()
        else:
            report("bench", False, "needs a CUDA GPU")
    print("\nall checks passed" if not failures else f"\n{len(failures)} mismatch(es): {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
