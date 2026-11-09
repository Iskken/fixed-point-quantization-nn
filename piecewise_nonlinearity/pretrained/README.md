# Pretrained float ResNet-20 (CIFAR-10)

These are the full-precision starting points for every quantization run (STE, staircase, EWGS, DSQ). Use them as they are and don't retrain them, so that all of us start from exactly the same weights.

| File | Validation acc. (5k) | Test acc. (10k) | Training time |
|---|---|---|---|
| `resnet20_cifar10_seed0.pt` | 91.66 | 91.17 | 56 min |
| `resnet20_cifar10_seed1.pt` | 91.64 | 91.16 | 55 min |
| `resnet20_cifar10_seed2.pt` | 91.86 | 91.23 | 55 min |

All three were trained with code commit `fc7f2bb` on an RTX 3050 Laptop GPU, using:

```
python -m piecewise_nonlinearity.train_float --seed N
```

The recipe follows the official EWGS code: SGD with momentum 0.9, learning rate 0.1 with cosine decay to 0, weight decay 1e-4, 400 epochs, batch size 256.

For comparison, EWGS reports 91.4 and DSQ 90.78 for their full-precision models. We train on 45k images and keep 5k aside for validation; EWGS trains on all 50k.

The per-epoch curves are in `../results/float/resnet20_cifar10_seed<N>.json`.

## Loading

```python
import torch
from piecewise_nonlinearity.models.resnet20 import ResNet20

ckpt = torch.load("piecewise_nonlinearity/pretrained/resnet20_cifar10_seed0.pt",
                  map_location="cpu", weights_only=True)
model = ResNet20()
model.load_state_dict(ckpt["model"])
model.eval()
print(ckpt["meta"])  # settings, accuracies, git commit, torch version, GPU
```

Inputs must be normalized the same way as in training. The simplest way is `piecewise_nonlinearity.datasets.get_loaders("cifar10")`. It normalizes with mean (0.4914, 0.4822, 0.4465) and std (0.2023, 0.1994, 0.2010), the values from the EWGS code.

To start a quantized model from one of these files, use `load_float_weights` in `models/quant_layers.py`.
