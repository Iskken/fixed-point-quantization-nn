import torch
import torchvision

"""
MNIST and SVHN as plain tensors, split into train / val / test.

Images stay uint8 (N, C, H, W) and are scaled to float in [0, 1] one batch
at a time, so a whole dataset fits on a small GPU (SVHN train is ~225 MB as
uint8, ~900 MB as float32). No mean/std normalisation: inputs in [0, 1]
need no integer bits at all, which keeps the input's fixed-point format
trivial to reason about.

The official test split is held out untouched; validation is carved out of
the official training split and is what early stopping and every
learning-rate or bit-allocation choice may look at.

Datasets are downloaded into data/ (gitignored) on first use.
"""

N_CLASSES = 10


def _raw_splits(name, root):
    if name == "mnist":
        train = torchvision.datasets.MNIST(root=root, train=True, download=True)
        test = torchvision.datasets.MNIST(root=root, train=False, download=True)
        # (N, 28, 28) -> (N, 1, 28, 28)
        return (train.data.unsqueeze(1), train.targets.long(),
                test.data.unsqueeze(1), test.targets.long())

    if name == "svhn":
        # torchvision already maps SVHN's digit "10" label to class 0
        train = torchvision.datasets.SVHN(root=f"{root}/svhn", split="train", download=True)
        test = torchvision.datasets.SVHN(root=f"{root}/svhn", split="test", download=True)
        return (torch.from_numpy(train.data), torch.from_numpy(train.labels).long(),
                torch.from_numpy(test.data), torch.from_numpy(test.labels).long())

    raise ValueError(f"Unknown dataset: {name}")


def load_image_dataset(name, root="data", val_fraction=0.1, seed=0, device="cpu"):
    """
    Returns a dict with uint8 images Xtr / Xva / Xte, int64 labels ytr / yva
    / yte, and metadata. `seed` only decides which training images become
    validation images; the test split is fixed.
    """
    X_train_full, y_train_full, X_test, y_test = _raw_splits(name, root)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X_train_full), generator=generator)
    n_val = int(round(val_fraction * len(X_train_full)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    return {
        "name": name,
        "Xtr": X_train_full[train_idx].to(device), "ytr": y_train_full[train_idx].to(device),
        "Xva": X_train_full[val_idx].to(device), "yva": y_train_full[val_idx].to(device),
        "Xte": X_test.to(device), "yte": y_test.to(device),
        "in_channels": X_test.shape[1],
        "image_size": X_test.shape[2],
        "n_classes": N_CLASSES,
    }


def to_float(images_uint8):
    """uint8 pixels -> float32 in [0, 1]."""
    return images_uint8.float() / 255.0


def iterate_batches(X, y, batch_size, shuffle=False, generator=None):
    """Yields (float images in [0, 1], labels) mini-batches."""
    n = len(X)
    order = torch.randperm(n, generator=generator, device="cpu").to(X.device) if shuffle \
        else torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield to_float(X[idx]), y[idx]
