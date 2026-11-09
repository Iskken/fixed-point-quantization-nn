# Setting up a machine for the experiments

We run experiments on two laptops: an RTX 3050 with 4 GB and an RTX 3050 with 6 GB, both on Linux. Their results are only comparable if both machines use the same software, data, validation split and float models. Follow these steps once per machine.

## 1. Code

```
git clone https://github.com/Iskken/fixed-point-quantization-nn.git
cd fixed-point-quantization-nn
git checkout directory_preparation   # or main, whichever the team is on
```

## 2. Python environment (exact versions)

Use Python 3.11 and a fresh virtual environment in the repository root:

```
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r piecewise_nonlinearity/requirements-lock.txt
```

If `python3.11` is not installed, `uv venv --python 3.11 .venv` downloads it. Then install with `uv pip install -r piecewise_nonlinearity/requirements-lock.txt`.

If you already have a `.venv` with other versions, create a new one; don't mix them.

## 3. Data

```
.venv/bin/python -m piecewise_nonlinearity.datasets
```

This downloads CIFAR-10 and CIFAR-100 into `piecewise_nonlinearity/data/` (about 680 MB) and checks their MD5 checksums.

## 4. Check the machine

```
.venv/bin/python -m piecewise_nonlinearity.check_env --bench
```

Every line must say **OK** or **INFO**, and the last line must say "all checks passed". The script checks:
- the Python, torch, torchvision, NumPy, Pillow, CUDA and cuDNN versions;
- the dataset checksums;
- the validation split fingerprint;
- the three float models;
- that the seed-0 float model reaches its recorded validation accuracy of 91.66.

The `--bench` lines give this GPU's speed; send them to the team, so we can split the runs between the laptops.

## 5. Tests

```
.venv/bin/python -m pytest piecewise_nonlinearity/tests -q
```

All tests must pass.

## Rules for runs that will be compared

- **Same commit on both laptops.** Run `git pull` before every batch of runs. Every result file records the commit.
- **One method's tuning stays on one machine.**
- **Each seed's methods stay on one machine.** All methods for a given seed run on the same laptop, so comparisons are never across machines. Every result file records the GPU.
- **Push result files after each batch.** They are small JSON files in `piecewise_nonlinearity/results/`.
- **If the versions ever need to change, change them for both laptops at once.** Update `requirements-lock.txt` and the expected values in `check_env.py` in the same commit.
