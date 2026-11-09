"""
CIFAR-10 and CIFAR-100 for the piecewise / EWGS / DSQ comparison.

The datasets are downloaded once into piecewise_nonlinearity/data/ (git-ignored) with:
    python -m piecewise_nonlinearity.datasets

get_loaders() splits the official training set into training and validation images with a fixed
seed (validation is used for every choice; the official test set only for the final numbers), and
uses the standard CIFAR augmentation for training: random crop with padding 4 and horizontal flip.
"""

import hashlib
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    # CIFAR-10 values copied from the official EWGS code (CIFAR10/train_fp.py), for a faithful reproduction
    "cifar10": dict(cls=datasets.CIFAR10, n_classes=10,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100": dict(cls=datasets.CIFAR100, n_classes=100,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
}


def download(name):
    """Download the training and test split (torchvision checks the archive's MD5)."""
    cls = DATASETS[name]["cls"]
    return cls(DATA_DIR, train=True, download=True), cls(DATA_DIR, train=False, download=True)


def split_indices(n_total, val_fraction=0.1, seed=0):
    """The fixed train/validation split of the official training set, as (train, validation) indices."""
    order = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(round(val_fraction * n_total))
    return order[n_val:], order[:n_val]


def split_fingerprint(val_idx):
    """Short hash of the validation indices: equal on two machines means the same validation images."""
    return hashlib.sha256(",".join(map(str, val_idx)).encode()).hexdigest()[:16]


def _transforms(name):
    """(training transform with augmentation, evaluation transform without)."""
    spec = DATASETS[name]
    normalize = transforms.Normalize(spec["mean"], spec["std"])
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, eval_tf


def _loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def get_loaders(name, batch_size=256, val_fraction=0.1, seed=0, num_workers=4):
    """Training, validation and test loaders; the train/validation split is fixed by `seed`."""
    spec = DATASETS[name]
    train_tf, eval_tf = _transforms(name)

    # two views of the same training images: augmented for training, plain for validation
    train_full = spec["cls"](DATA_DIR, train=True, transform=train_tf)
    eval_full = spec["cls"](DATA_DIR, train=True, transform=eval_tf)
    test = spec["cls"](DATA_DIR, train=False, transform=eval_tf)

    train_idx, val_idx = split_indices(len(train_full), val_fraction, seed)
    train, val = Subset(train_full, train_idx), Subset(eval_full, val_idx)

    return (_loader(train, batch_size, True, num_workers), _loader(val, batch_size, False, num_workers),
            _loader(test, batch_size, False, num_workers))


def get_bn_loader(name, batch_size=256, val_fraction=0.1, seed=0, num_workers=4):
    """
    The training images of the same fixed split, without augmentation and in a fixed order: the data for
    BatchNorm re-estimation (models/quant_layers.reestimate_bn). Validation and test images are never in it.
    """
    spec = DATASETS[name]
    _, eval_tf = _transforms(name)
    eval_full = spec["cls"](DATA_DIR, train=True, transform=eval_tf)
    train_idx, _ = split_indices(len(eval_full), val_fraction, seed)
    return _loader(Subset(eval_full, train_idx), batch_size, False, num_workers)


if __name__ == "__main__":
    for name in DATASETS:
        train, test = download(name)
        print(f"{name}: {len(train)} training and {len(test)} test images, "
              f"{len(train.classes)} classes, in {DATA_DIR}")
