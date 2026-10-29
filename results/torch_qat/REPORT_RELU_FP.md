# ReLU float baseline: learning rate 0.01 vs 0.003

Written for: professors reviewing the QAT work.

## Why these two learning rates

The ReLU float network is the starting point for all later ReLU QAT and hard STE runs, so it has to be trained well first. Learning rate 0.01 is the rate used for the tanh float run, so it was the natural first try. It showed a noisy validation curve and a validation error that rose late in training. Learning rate 0.003 was added to test whether the noise came from a step size that was too large. Both runs use the same 5 seeds (same data, split and initial weights per seed), full-batch gradient descent, 100,000 epochs, architecture [4,32,32,16,1].

## Figures

Left: lr 0.01. Right: lr 0.003. Each shows training loss (left panel) and validation loss (right panel), float inference. Dots on the validation panel mark the best-validation epoch of each seed, which is the checkpoint used as Start B.

- `relu_lr0.01_100k/fig_relu_fp_train_vs_val_lr0.01.png`
- `relu_lr0.003/fig_relu_fp_train_vs_val_lr0.003.png`

## What the figures show

**Learning rate 0.01**
- Training loss falls fast, passes a mild plateau at about 0.5 to 0.3, and keeps falling to about 0.05 to 0.07 at 100k epochs.
- Validation loss reaches its minimum (about 0.19 to 0.23) between epochs 29k and 65k, then creeps slightly upward while training loss keeps falling. This is overfitting.
- The curves are thick, noisy bands rather than lines throughout the run. The validation band is about ±0.03 wide.

**Learning rate 0.003**
- Training and validation loss descend more slowly and are visibly smoother for the first 50k to 60k epochs.
- Seeds 1 and 2 settle at validation about 0.20 to 0.22. Seed 4 is the slowest (validation still about 0.35 at the end) and the highest.
- Seeds 0, 3 and 4 develop a thick band late in training, from about epoch 60k to 80k, and the band grows. Seed 0's validation turns up near epoch 80k.
- Best validation is worse than at 0.01 (mean 0.257 against 0.206), because the slower rate has not finished descending in several seeds.

## Why the curves fluctuate

Each plotted point is the loss after one full-batch update, so the band is not batch sampling noise. The data are the same every epoch. The fluctuation is the weights jumping between epochs, which comes from the step being too large for the local curvature of the loss.

A ReLU network has a loss surface that is piecewise-quadratic. Each hidden unit switches on or off depending on the sign of its pre-activation. When a step moves a pre-activation across zero for some samples, the set of active units changes and the gradient changes abruptly. With a large step, successive updates overshoot, the weights bounce across the valley, and the loss alternates between high and low values. Plotted over thousands of epochs, this shows as a thick band. Validation is affected more than training because the network is fitted to the training points, and a small weight jump moves the output more at unseen points.

I have not tested this explanation directly (for example by counting unit flips per epoch). It fits what the curves and the gradient norms show, but it is a hypothesis.

## Why 0.01 fluctuates from early on and 0.003 is smooth for longer

The size of the oscillation grows with the step size. At 0.01 the step is above the stability limit of the sharper directions of the loss surface almost from the point where the network leaves the plateau. The weights hover around the minimum with persistent jitter, which shows from about epoch 15k onwards.

At 0.003 the step is below that limit while the loss surface is still relatively flat, so the descent is smooth. The curves stay smooth until the network moves into a sharper region. The weights then grow and the surface gets sharper (training loss falls to about 0.1, and the gradient norm in seeds 0, 3 and 4 jumps from about 0.1 to 5 to 7). At that point even 0.003 is too large and oscillations start, then grow. This is why the smooth 0.003 curves develop bands late, and why the band in those seeds gets larger instead of smaller. The lower rate delays the instability. It does not remove it.

So the two runs have the same kind of fluctuation. At 0.01 it starts early and stays about the same width. At 0.003 it starts late.

## Why the final weights come from lr 0.01

1. **Better validation.** Mean best validation MSE is 0.206 at lr 0.01 and 0.257 at lr 0.003. At 0.01 every seed reached its minimum well before the end of the run. At 0.003 several seeds were still descending or had begun to oscillate, so the checkpoint is a worse fit.
2. **The fluctuation does not affect the checkpoint.** The start weights are the best-validation checkpoint of each seed, not the final epoch. The best epochs at lr 0.01 are 58,740, 47,089, 28,847, 48,203 and 64,967. These lie before the validation curve rises again and, even inside the band, are a point of low validation error. A smoother curve at lr 0.003 gives no advantage if the best point it reaches is worse.
3. **Consistency with the tanh experiments.** The tanh float run used lr 0.01, so using the same rate keeps the preparation of the starting point the same.
4. **Seed 4 at 0.003** is far from converged (validation about 0.35), which would make its QAT result partly a measure of an undertrained start.

Caveat: the choice of 0.01 is based on validation error of the float network, not on QAT results. No QAT run was done from the 0.003 weights, so I cannot say whether a smoother start would have given a better quantized result. Only the lr 0.01 / 100k checkpoints (`relu_lr0.01_100k/checkpoints/`) were used for the 2×2, grid and Optuna runs.
