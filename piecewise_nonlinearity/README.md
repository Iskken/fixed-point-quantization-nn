# Piecewise-linear staircase vs. EWGS and DSQ

This folder is a self-contained workspace for comparing our piecewise-linear staircase quantizer with two published methods, **EWGS** (Lee et al., CVPR 2021) and **DSQ** (Gong et al., ICCV 2019). The main benchmark is the one both papers share, **ResNet-20 on CIFAR-10**.

It does not import from or write to `experiments/`, `notebooks/`, `results/` or `src/` at the repository root. Code we want to reuse from there, such as `src/quantization/torch_stair.py`, is copied in and adapted here, so this folder can be read and run on its own.

## What we test: the pure staircase

During training the network computes with the soft staircase S_{α,β}(x), not with exact rounding. α and β set the shape of each step. An α schedule makes S approach exact rounding by the end of training.

We do not use the "hybrid" mode (exact rounding forward, staircase slopes backward). With Adam, a backward-only change like that trains almost exactly like the straight-through estimator (STE). That result is Schoenbauer et al. 2024, "Custom Gradient Estimators are Straight-Through Estimators in Disguise". The pure staircase changes the forward pass, so the result does not apply to it.

How this relates to the two baselines:
- **EWGS and DSQ** both keep a hard forward pass and change only the gradient.
- **QuantNet** (Yang et al., CVPR 2019) is the closest prior work: a soft sigmoid staircase sharpened by a temperature.

What our earlier MLP study found ([STUDY_LOG.md](../results/torch_qat/study_claude/STUDY_LOG.md); ReLU MLP regression, Adam, fixed-point formats):
- **β = 1 fails with the exact backward.** With large α the quantized weights freeze, and the result is clearly worse than the STE.
- **The backward slopes matter.** Exact and slope-1 backward behaved very differently, so the switch is a real ablation.
- **One setting beat the STE:** exact backward, α = 2, β = 2, at the 8/4 format, on 20 fresh seeds. It tied at 6/4 and was slightly worse at 6/3, so the win is narrow.
- **Small α was enough.** The gap between the staircase forward and exact rounding at evaluation stayed small, so α does not have to grow huge.

## Rules for every experiment

- **Score every method with exact rounding at test time.** The fixed-point hardware rounds; it never computes the soft staircase.
- **Tune every method the same way.** The STE gets the same learning-rate search as the staircase.
- **Choose on validation, report on test.** Every choice uses the validation split; the test set is used only for the final numbers.
- **Tune on development seeds, claim on fresh seeds.** This is the same rule as in the MLP study. Report how many settings each method was tuned over, since the staircase has two extra knobs (α, β).
- **Re-estimate BatchNorm statistics before testing.** The staircase trains with S, so its BatchNorm statistics don't match exact rounding. Recompute them with exact rounding on a few training batches, the same way for every method, and report results with and without it.

## Plan

| Step | What | Where |
|---|---|---|
| 1 | Folder structure (this README) | this folder |
| 2 | Download CIFAR-10 and CIFAR-100 into `data/` | `datasets.py` |
| 3 | Train a float ResNet-20 (and later MobileNetV2) once per seed; every method starts from it | `models/`, `train_float.py` |
| 4 | Shared quantizer (clip to [l, u], uniform levels) and the STE baseline | `quantizers/base.py`, `quantizers/ste.py` |
| 5 | Our pure staircase: soft forward S_{α,β} in training, exact rounding at test time; backward either its exact slopes (main) or slope 1 (ablation); α schedule, β fixed per run | `quantizers/piecewise.py` |
| 6 | Sanity tests (see below) | `tests/` |
| 7 | **Decisive experiment**, described below | `train_qat.py` |
| 8 | EWGS and DSQ | `quantizers/ewgs.py`, `quantizers/dsq.py` |
| 9 | Reproduce the paper numbers below, to show our EWGS and DSQ are implemented correctly | `train_qat.py`, `configs/` |
| 10 | Main comparison: STE, staircase, EWGS and DSQ at 2/2, 3/3 and 4/4 on ResNet-20; MobileNetV2 at 4/4; CIFAR-100 if time allows | `train_qat.py`, `results/` |

Step 6, sanity tests:
- the staircase with α = β = 1 is the identity;
- at large α its forward pass approaches exact rounding;
- EWGS with δ = 0 equals the STE exactly;
- the gradient is zero outside the clipping range.

Step 7, decisive experiment: ResNet-20, Adam, short fine-tuning runs (about 30 epochs). Proposed setup:
- **Bit widths:** 4/4 (stable, and closest to where the MLP study found its win) and 2/2 (where EWGS and DSQ differ most from the STE).
- **Tuning, on one development seed:** the STE over the learning rate; the staircase with exact backward over learning rate × α ∈ {1.5, 2, 3} × β ∈ {1, 2, 3}; the slope-1 backward at the best α and β.
- **Comparison:** the best setting of each method on 3 seeds, for weights only, activations only, and both.
- **Logged per run:** the gap between staircase-forward and exact-rounding validation accuracy, and the share of weights that change level per epoch.
- **Why it comes first:** it shows whether α and β work through the forward curve or the backward slopes, and whether the effect comes from weights or activations. That decides how we word the abstract.

ImageNet is out of scope: it does not fit our GPU (4 GB) or our timeline.

## Structure

```
piecewise_nonlinearity/
├── README.md
├── datasets.py           CIFAR-10 / CIFAR-100 loaders and download
├── quantizers/
│   ├── base.py           shared quantizer: clip to [l, u], uniform levels
│   ├── ste.py            straight-through estimator (baseline)
│   ├── piecewise.py      our pure staircase S_{α,β}, backward switch, α schedule
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
├── data/                 CIFAR-10 and CIFAR-100 (git-ignored)
├── pretrained/           final float models, ~1 MB each (committed, so everyone starts from the same weights)
└── checkpoints/          resume checkpoints with optimizer state (git-ignored)
```

Run everything from the repository root as a module, for example:

```
python -m piecewise_nonlinearity.datasets
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

- **EWGS (ResNet-20):**
  - Adam, learning rate 1e-3 with cosine decay, weight decay 1e-4, 400 epochs, batch size 256.
  - Quantizer parameters (l, u, output scale): Adam with learning rate 1e-5 and no weight decay.
  - δ starts at 0 and is updated every 10 epochs.
  - l and u start at ±3σ(w) for weights, and at 0 and 3σ(a)/√(1−2/π) for activations.
- **ResNet-20 and float pre-training, from the official EWGS code** (`CIFAR10/` in github.com/cvlab-yonsei/EWGS):
  - option-A shortcut (parameter-free, zero padding), post-activation blocks, an extra BatchNorm1d before the final linear layer, Kaiming initialization;
  - quantized: the 18 convolutions inside the blocks (weights and input activations); the first conv, the final linear layer and all BatchNorms stay in float;
  - each quantized conv has a learned output scale, initialized to mean|float output| / mean|quantized output|;
  - float training: SGD with momentum 0.9, learning rate 0.1 with cosine decay to 0, weight decay 1e-4, 400 epochs, batch size 256, on all 50k training images; QAT loads the last float checkpoint;
  - δ is estimated on 3 training batches, with up to 50 Hutchinson samples per tensor;
  - input normalization std (0.2023, 0.1994, 0.2010), which `datasets.py` copies for CIFAR-10.
  - We train on 45k images and keep 5k for validation, so our full-precision accuracy may come out slightly below theirs.
- **DSQ:**
  - α starts at 0.2 and is kept in (0, 0.5), with k ≤ 1000.
  - The clipping bounds come either from moving-average statistics or from backpropagation.
  - The training forward pass is hard: tanh, then the sign function (Algorithm 1).
  - Binarized models use the type-A (parameter-free) shortcut.
- **Both:** the first and last layers are not quantized unless stated otherwise.

400 epochs per run will not fit our schedule for every method, bit width and seed. Decide the fine-tuning budget once and use it for all methods; report it next to any comparison with the paper numbers.

## Compute budget (RTX 3050 Laptop, 4 GB)

Measured with synthetic data, fp32 and a simple 4-bit STE quantizer. The real quantizers will be slower, so treat these as lower bounds.

| Model | QAT seconds per epoch |
|---|---|
| ResNet-20 | 14.7 |
| MobileNetV2 (CIFAR version) | 94.7, at batch size 128 (batch 256 does not fit) |

The data loader needs about 4 s per epoch, so it is not the bottleneck.

## Reference code (not committed)

We read the authors' code as a reference and write our own implementations; we never import or copy it. The repositories are cloned into `reference/`, which is git-ignored. They are kept out of our repo because the EWGS code is GPL-3.0, the unofficial DSQ code has no license, and both are separate git repositories. To get the same versions we used, run from `piecewise_nonlinearity/`:

```
git clone https://github.com/cvlab-yonsei/EWGS.git reference/EWGS
git -C reference/EWGS checkout 56c654c
git clone https://github.com/ricky40403/DSQ.git reference/DSQ_unofficial
git -C reference/DSQ_unofficial checkout f06df7f
```

- **EWGS** (official): the CIFAR-10 quantizer is in `CIFAR10/custom_modules.py`, and the training scripts are `CIFAR10/train_fp.py` and `train_quant.py`.
- **DSQ** (unofficial, ImageNet only; the authors released no training code): a secondary reference. The paper's equations and Algorithm 1 are the primary source.

## Conventions

- Every number we report comes from a saved JSON file in `results/`, written by the script that produced it.
- Datasets and checkpoints stay local (`.gitignore` in this folder); summaries and configs are committed.
- Push code and result summaries as soon as a run finishes.

## Status

- [x] Folder structure and README
- [x] CIFAR-10 and CIFAR-100 download
- [x] Float ResNet-20, seeds 0-2: test 91.17 / 91.16 / 91.23 (`pretrained/`)
- [x] Shared quantizer + STE
- [x] Pure staircase
- [x] Sanity tests (23 passing)
- [ ] Decisive experiment
- [ ] EWGS
- [ ] DSQ
- [ ] Paper reproductions
- [ ] Main comparison
