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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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


def _loader(ds, batch_size, shuffle, num_workers, generator=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      generator=generator)


def get_loaders(name, batch_size=256, val_fraction=0.1, seed=0, num_workers=4, generator=None):
    """
    Training, validation and test loaders; the train/validation split is fixed by `seed`.
    `generator` (optional) drives the training loader's shuffling and its workers' random seeds, so the
    order and the augmentation of the training images depend on it alone (with num_workers > 0).
    """
    spec = DATASETS[name]
    train_tf, eval_tf = _transforms(name)

    # two views of the same training images: augmented for training, plain for validation
    train_full = spec["cls"](DATA_DIR, train=True, transform=train_tf)
    eval_full = spec["cls"](DATA_DIR, train=True, transform=eval_tf)
    test = spec["cls"](DATA_DIR, train=False, transform=eval_tf)

    train_idx, val_idx = split_indices(len(train_full), val_fraction, seed)
    train, val = Subset(train_full, train_idx), Subset(eval_full, val_idx)

    return (_loader(train, batch_size, True, num_workers, generator), _loader(val, batch_size, False, num_workers),
            _loader(test, batch_size, False, num_workers))


def get_train_eval_loader(name, n, batch_size=256, val_fraction=0.1, seed=0, num_workers=4):
    """
    n training images of the same fixed split, without augmentation and in a fixed order: for measuring the
    fit to the training data the way validation is measured (train_qat.py does it every epoch). They are the
    first n training indices of split_indices, which come in random order, so they are a random sample of
    the training images; validation and test images are never among them. The loader has its own generator,
    so iterating it never draws from the global random state.
    """
    spec = DATASETS[name]
    _, eval_tf = _transforms(name)
    eval_full = spec["cls"](DATA_DIR, train=True, transform=eval_tf)
    train_idx, _ = split_indices(len(eval_full), val_fraction, seed)
    return _loader(Subset(eval_full, train_idx[:n]), batch_size, False, num_workers,
                   generator=torch.Generator().manual_seed(0))


class SeededAugmentation(Dataset):
    """
    The training augmentation (random crop with 4 pixels of zero padding, random horizontal flip, then
    normalization), with each image's random draws fixed by `seed` and the image's index. Every pass gives
    the same images, on any machine and with any number of workers. The distribution is the same as
    transforms.RandomCrop(32, padding=4) + RandomHorizontalFlip(): crop offsets uniform in 0..8, flip with
    probability 0.5, padding with black pixels before normalization.
    `base` must return (ToTensor image, label).
    """

    def __init__(self, base, indices, mean, std, seed=0):
        self.base, self.indices, self.seed = base, list(indices), seed
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.indices)

    def draws(self, i):
        """(top, left, flip) for item i."""
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.indices[i])
        top, left = torch.randint(0, 9, (2,), generator=g).tolist()
        return top, left, bool(torch.rand((), generator=g) < 0.5)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        top, left, flip = self.draws(i)
        x = F.pad(x, (4, 4, 4, 4))[:, top:top + 32, left:left + 32]
        if flip:
            x = x.flip(-1)
        return (x - self.mean) / self.std, y


def get_bn_loader(name, batch_size=256, val_fraction=0.1, seed=0, num_workers=4, aug_seed=0):
    """
    The data for BatchNorm re-estimation (models/quant_layers.reestimate_bn): the training images of the
    same fixed split, in a fixed order, with the training augmentation seeded per image (SeededAugmentation).
    Validation and test images are never in it.

    The augmentation matters. During training, BatchNorm always normalized augmented batches (random crops
    with black padding, flips), and every later layer adapted to that normalization. Statistics of clean
    images normalize differently, even for images the network has never seen: re-estimating on clean images
    lowered the float models' accuracy by about 4.5 points (91.63 -> 87.17, seeds 0-2, on half of the
    validation set), the same for clean training images and for unseen clean images. With the training
    augmentation it stays unchanged (91.71). Measured 2026-11-09. So re-estimation uses the input
    distribution BatchNorm saw during training.
    """
    spec = DATASETS[name]
    base = spec["cls"](DATA_DIR, train=True, transform=transforms.ToTensor())
    train_idx, _ = split_indices(len(base), val_fraction, seed)
    return _loader(SeededAugmentation(base, train_idx, spec["mean"], spec["std"], seed=aug_seed),
                   batch_size, False, num_workers)


if __name__ == "__main__":
    for name in DATASETS:
        train, test = download(name)
        print(f"{name}: {len(train)} training and {len(test)} test images, "
              f"{len(train.classes)} classes, in {DATA_DIR}")
