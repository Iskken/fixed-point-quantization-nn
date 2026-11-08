# Piecewise-linear staircase vs. EWGS and DSQ

This folder is a self-contained workspace for comparing our piecewise-linear staircase quantizer with two published methods, **EWGS** (Lee et al., CVPR 2021) and **DSQ** (Gong et al., ICCV 2019), on the benchmark that both papers share: **ResNet-20 on CIFAR-10**.

It does not import from or write to `experiments/`, `notebooks/`, `results/` or `src/` at the repository root. Code we want to reuse from there, such as `src/quantization/torch_stair.py`, is copied in and adapted here, so this folder can be read and run on its own.

## Plan

| Step | What | Where |
|---|---|---|
| 1 | Folder structure (this README) | this folder |
| 2 | Download CIFAR-10 into `data/` | `datasets.py` |
| 3 | Train a float ResNet-20 once per seed; every method starts from it | `models/resnet20.py`, `train_float.py` |
| 4 | Shared quantizer and STE baseline: same forward pass for every method, only the backward rule differs | `quantizers/base.py`, `quantizers/ste.py` |
| 5 | Our piecewise staircase, usable from other modules | `quantizers/piecewise.py` |
| 6 | EWGS and DSQ | `quantizers/ewgs.py`, `quantizers/dsq.py` |
| 7 | Sanity tests: piecewise (slopes 1) and EWGS (δ = 0) must equal the STE exactly | `tests/` |
| 8 | Easiest experiments first: short fine-tuning runs at 4/4 to check that everything trains | `train_qat.py` |
| 9 | Reproduce the paper numbers below, to show our EWGS and DSQ are implemented correctly | `train_qat.py`, `configs/` |
| 10 | Main comparison: STE, piecewise, EWGS, DSQ at 2/2, 3/3, 4/4 under one protocol | `train_qat.py`, `results/` |

ImageNet is out of scope: it does not fit our GPU (4 GB) or our timeline.

## Structure

```
piecewise_nonlinearity/
├── README.md
├── datasets.py           CIFAR-10 loaders and download
├── quantizers/
│   ├── base.py           shared quantizer: clip to [l, u], exact rounding forward
│   ├── ste.py            straight-through estimator (baseline)
│   ├── piecewise.py      our staircase: "hybrid" and "staircase" modes, alpha/beta
│   ├── ewgs.py           EWGS backward rule and Hessian-based delta
│   └── dsq.py            DSQ tanh soft quantizer with learnable alpha, l, u
├── models/
│   ├── quant_layers.py   QConv2d / QLinear with a pluggable quantizer
│   └── resnet20.py       ResNet-20 for CIFAR-10
├── train_float.py        float pre-training (shared starting point)
├── train_qat.py          QAT fine-tuning, one entry point for all methods
├── configs/              one config file per experiment
├── results/              per-run JSON summaries (committed)
├── tests/                sanity checks for the quantizers
├── data/                 CIFAR-10 (git-ignored)
└── checkpoints/          model weights (git-ignored)
```

Run everything from the repository root as a module, for example:

```
python -m piecewise_nonlinearity.train_float --seed 0
python -m piecewise_nonlinearity.train_qat --method piecewise --bits 4/4 --seed 0
python -m pytest piecewise_nonlinearity/tests
```

## Numbers to reproduce (ResNet-20, CIFAR-10, top-1 test accuracy)

| Paper | Setting (W/A) | Method | Accuracy | Full precision |
|---|---|---|---|---|
| EWGS, Tab. 4–5 | 1/1 | STE | 84.7 | 91.4 |
| EWGS, Tab. 4–5 | 1/1 | EWGS | 85.6 | 91.4 |
| EWGS, Tab. 5 | 1/1, DoReFa, δ fixed 0.01 | EWGS | 85.9 | 91.4 |
| EWGS, Tab. 5 | 1/32, DoReFa, all layers quantized | EWGS | 90.3 | 91.4 |
| DSQ, Tab. 3, 6 | 1/1 | DSQ | 84.11 | 90.78 |
| DSQ, Tab. 3, 6 | 1/32 | DSQ | 90.24 | 90.78 |
| DSQ, Tab. 4 | 2-bit | learned α, l, u | 88.44 | 90.78 |

The two papers start from different full-precision models (91.4 vs 90.78), so compare the drop from full precision, not the raw accuracy. Neither paper reports 3- or 4-bit results on CIFAR-10; for those settings we run every method ourselves.

## Training settings from the papers

- **EWGS (ResNet-20):** Adam, learning rate 1e-3 with cosine decay, weight decay 1e-4, 400 epochs, batch size 256. Quantizer parameters (l, u, output scale) use Adam with learning rate 1e-5 and no weight decay. δ starts at 0 and is updated every 10 epochs. l, u start at ±3σ(w) for weights, and at 0 and 3σ(a)/√(1−2/π) for activations.
- **DSQ:** α starts at 0.2 and is kept in (0, 0.5) with k ≤ 1000. The clipping bounds come either from moving-average statistics or from backpropagation. Binarized models use the type-A (parameter-free) shortcut.
- **Both:** the first and last layers are not quantized unless stated otherwise.

400 epochs per run will not fit our schedule for every method, bit width and seed. Decide the fine-tuning budget once and use it for all methods; report it next to any comparison with the paper numbers.

## Conventions

- Every number we report comes from a saved JSON file in `results/`, written by the script that produced it.
- Datasets and checkpoints stay local (`.gitignore` in this folder); summaries and configs are committed.
- Push code and result summaries as soon as a run finishes.

## Status

- [x] Folder structure and README
- [ ] CIFAR-10 download
- [ ] Float ResNet-20
- [ ] Shared quantizer + STE
- [ ] Piecewise staircase
- [ ] EWGS
- [ ] DSQ
- [ ] Sanity tests
- [ ] First short runs
- [ ] Paper reproductions
- [ ] Main comparison
