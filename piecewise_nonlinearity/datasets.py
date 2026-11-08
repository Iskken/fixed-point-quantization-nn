"""
CIFAR-10 and CIFAR-100 for the piecewise / EWGS / DSQ comparison.

The datasets are downloaded once into piecewise_nonlinearity/data/ (git-ignored) with:
    python -m piecewise_nonlinearity.datasets

get_loaders() splits the official training set into training and validation images with a fixed
seed (validation is used for every choice; the official test set only for the final numbers), and
uses the standard CIFAR augmentation for training: random crop with padding 4 and horizontal flip.
"""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    "cifar10": dict(cls=datasets.CIFAR10, n_classes=10,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(cls=datasets.CIFAR100, n_classes=100,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
}


def download(name):
    """Download the training and test split (torchvision checks the archive's MD5)."""
    cls = DATASETS[name]["cls"]
    return cls(DATA_DIR, train=True, download=True), cls(DATA_DIR, train=False, download=True)


def get_loaders(name, batch_size=256, val_fraction=0.1, seed=0, num_workers=4):
    """Training, validation and test loaders; the train/validation split is fixed by `seed`."""
    spec = DATASETS[name]
    normalize = transforms.Normalize(spec["mean"], spec["std"])
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    # two views of the same training images: augmented for training, plain for validation
    train_full = spec["cls"](DATA_DIR, train=True, transform=train_tf)
    eval_full = spec["cls"](DATA_DIR, train=True, transform=eval_tf)
    test = spec["cls"](DATA_DIR, train=False, transform=eval_tf)

    order = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(round(val_fraction * len(order)))
    train, val = Subset(train_full, order[n_val:]), Subset(eval_full, order[:n_val])

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

    return loader(train, True), loader(val, False), loader(test, False)


if __name__ == "__main__":
    for name in DATASETS:
        train, test = download(name)
        print(f"{name}: {len(train)} training and {len(test)} test images, "
              f"{len(train.classes)} classes, in {DATA_DIR}")
