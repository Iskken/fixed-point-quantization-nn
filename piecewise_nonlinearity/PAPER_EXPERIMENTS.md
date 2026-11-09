# Experiments in the EWGS and DSQ papers

Every experiment both papers report, with its settings, to help us choose step 9's reproduction targets (see PLAN.md, change 6). Numbers are top-1 accuracy in %. They are copied from the papers:
- EWGS: Lee, Kim, Ham, "Network Quantization with Element-wise Gradient Scaling", CVPR 2021;
- DSQ: Gong et al., "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks", ICCV 2019.

The last column says whether an experiment fits our setup: ResNet-20 on CIFAR-10, a 4 GB GPU, and our shared quantizer with values in [0, 1], weights mapped to [−1, 1], learned l and u, and first and last layers in float.
- ✅ fits as is.
- ⚠️ possible, but needs extra code or has an unknown setting.
- ❌ out of reach.

Notation:
- **W/A** = weight bits / activation bits.
- **†** = all layers quantized, including the first and last.
- **‡** = EWGS scaling factor δ fixed to 0.01.
- **\*** = reproduced by the EWGS authors.

---

## EWGS

### Training settings (Section 4.1, and the official code for CIFAR-10)

| Setting | ResNet-20, CIFAR-10 | ResNet-18, ImageNet | ResNet-34, ImageNet | MobileNetV2, ImageNet |
|---|---|---|---|---|
| Float start | Trained by the authors (91.4) | PyTorch pretrained | PyTorch pretrained | PyTorch pretrained, PROFIT's network structure |
| Optimizer (weights) | Adam | SGD | SGD | SGD |
| Learning rate (weights) | 1e-3 | 1e-2 | 1e-2 | 5e-3 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 | 4e-5 |
| Epochs | 400 | 100 | 100 | 100 |
| Batch size | 256 | 256 | 256 | 100 |
| Quantizer parameters (l, u, output scale) | Adam, 1e-5, no weight decay | same | same | same |
| Learning-rate schedule | cosine | cosine | cosine | cosine |
| δ updated every | 10 epochs | 1 epoch | 1 epoch | 1 epoch |
| Quantized layers | all but first and last | all but first and last | all but first and last | all († in the tables) |

Shared by all four models:
- **Initialization:** weights l, u = ±3σ(w); activations l = 0 and u = 3σ(a)/√(1−2/π); output scale = mean\|float output\| / mean\|quantized output\|.
- **δ:** starts at 0 and is set from the Hessian trace (Hutchinson's method), using G = 3σ of the gradient.

From the official CIFAR-10 code (`CIFAR10/train_fp.py`, `train_quant.py`):
- **Float training:** SGD with momentum 0.9, learning rate 0.1 with cosine decay, 400 epochs, batch size 256, on all 50k images.
- **Network:** option-A ResNet-20 with a BatchNorm1d before the final layer.
- **Reported score:** test accuracy of the **last** checkpoint.
- **Hessian trace:** δ is estimated on 3 batches, with up to 50 Hutchinson samples.

The momentum used on ImageNet is not stated.

### Experiments

| # | Source | Dataset, model | W/A | What is compared | EWGS result | Float | Fits us? |
|---|---|---|---|---|---|---|---|
| E1 | Table 1 | ImageNet, ResNet-18 | 1/1, 1/2, 2/2, 3/3, 4/4, 1/32, 2/32 | EWGS vs XNOR, PACT, LQ-Net, QIL, QuantNet, DSQ, LSQ, LSQ+, IR-Net | 55.3, 64.4, 67.0, 69.7, 70.6, 67.3, 69.6 | 69.9 | ❌ ImageNet |
| E2 | Table 2 | ImageNet, ResNet-34 | 1/1, 1/2, 2/2, 3/3, 4/4, 1/32 | EWGS vs LSQ, ABC-Net, LQ-Net, QIL, DSQ, IR-Net | 61.5, 69.6, 71.4, 73.3, 73.9, 72.2 | 73.3 | ❌ ImageNet |
| E3 | Table 3 | ImageNet, MobileNetV2 | 4/4† | EWGS vs PACT (61.4), DSQ (64.8), PROFIT (71.6†) | 70.3† | 71.9 | ❌ ImageNet |
| E4 | Table 4 | CIFAR-10, ResNet-20 | 1/1 | How δ is set (7 variants, see below) | 85.6 (best variant) | 91.4 | ✅ |
| E5a | Table 5 | ImageNet, ResNet-18 | 1/1, 1/32 | STE vs EWGS | STE 54.6 → 55.3; STE 66.3 → 67.3 | 69.9 | ❌ ImageNet |
| E5b | Table 5 | ImageNet, MobileNetV2 | 4/4† | STE vs EWGS, in EWGS's quantizer and in PROFIT's | STE 69.2 → 70.3; PROFIT STE 69.2\* → 70.0‡ | 71.9 | ❌ ImageNet |
| **E5c** | **Table 5** | **CIFAR-10, ResNet-20** | **1/1** | **STE vs EWGS (EWGS's own quantizer)** | **STE 84.7 → EWGS 85.6** | **91.4** | **✅ direct match** |
| E5d | Table 5 | CIFAR-10, ResNet-20 | 1/1 | STE vs EWGS in the DoReFa quantizer | STE 84.9\* → 85.9‡ | 91.4 | ⚠️ needs a DoReFa quantizer |
| E5e | Table 5 | CIFAR-10, ResNet-20 | 1/32† | STE vs EWGS in the DoReFa quantizer, all layers | STE 89.7\* → 90.3‡ | 91.4 | ⚠️ DoReFa, plus first and last layers quantized |
| E6 | Figure 3 | ImageNet, ResNet-18 | 1/1 | δ over training, and per layer | analysis, no accuracy | — | ❌ |
| E7 | Figure 4 | ImageNet, ResNet-18 | 1/1, 1/32 | Training loss and accuracy curves, STE vs EWGS | curves only | — | ❌ |

**E4 in detail** (EWGS Table 4; ResNet-20, CIFAR-10, 1/1; float 91.4):

| How δ is set | Accuracy |
|---|---|
| Eq. 10 with G = 3σ(G_xq) (the paper's method) | 85.6 |
| Eq. 10 with G = max\|G_xq\| | 85.5 |
| Eq. 10 with G = E\|g_xq\| | 83.1 |
| Fixed δ = 1e-1 | 60.9 |
| Fixed δ = 1e-3 | 85.3 |
| Fixed δ = 1e-5 | 85.0 |
| Fixed δ = 0 (= STE) | 84.7 |

---

## DSQ

### Training settings (Section 4.1)

| Setting | Value |
|---|---|
| Quantized layers | all conv and fully-connected layers except the first and last |
| Binarized models | parameter-free type-A shortcut, plus BNN's activation-function replacement |
| Start | "fine-tuning from a pre-train model and training from scratch"; which one was used for each result is not stated |
| Hyperparameters | "we follow the rules described in the origin papers" (PACT, ResNet, LQ-Nets); **optimizer, learning rate, epochs and batch size are not stated** |
| α | starts at 0.2, kept in (0, 0.5) with k ≤ 1000; learned, except fixed at 0.2 in the binarization experiments |
| Clipping l, u | "moving average statistics" or "optimization by backward propagation" |
| Training forward pass | hard: tanh φ, then the sign function, then dequantization (Algorithm 1) |

### Experiments

| # | Source | Dataset, model | W/A | What is compared | DSQ result | Float | Fits us? |
|---|---|---|---|---|---|---|---|
| D1 | Table 2 | CIFAR-10, ResNet-20 | not stated | Learned α per layer | analysis | — | ✅ as analysis |
| D2a | Table 3 | CIFAR-10, ResNet-20 | 1/1 | Binary 82.46, Binary DSQ 83.80, **Piecewise DSQ 84.11** | 84.11 | 90.84 | ⚠️ ±1 activations and activation replacement differ from our {0, 1} quantizer |
| D2b | Table 3 | CIFAR-10, ResNet-20 | 1/32 | Binary 90.11, **Binary DSQ 90.24**, Piecewise DSQ 90.03 | 90.24 | 90.84 | ⚠️ close: our 1-bit weights are also ±1, but the binarization details are unclear |
| **D3** | **Table 4** | **CIFAR-10, ResNet-20** | **2-bit** (W/A not stated) | **Standard quantization vs DSQ ingredients (see below)** | **88.44** | not stated (90.78 in Table 6) | **✅ fits; W/A and schedule unknown** |
| D4 | Table 5 | not stated (ablation section, likely ResNet-20 on CIFAR-10) | 2-bit | PACT 88.24 vs PACT + DSQ | 90.11 | — | ⚠️ needs PACT; model not stated |
| D5a | Table 6 | CIFAR-10, VGG-Small | 1/1 | DSQ vs BNN (89.90), XNOR (89.80) | 91.72 | 91.65 | ⚠️ needs VGG-Small and the binarization variant |
| D5b | Table 6 | CIFAR-10, ResNet-20 | 1/1 | DSQ vs DoReFa (79.30) | 84.11 | 90.78 | ⚠️ same as D2a |
| D5c | Table 6 | CIFAR-10, ResNet-20 | 1/32 | DSQ vs DoReFa (90.00), LQ-Net (90.10) | 90.24 | 90.78 | ⚠️ same as D2b |
| D6a | Table 7 | ImageNet, ResNet-18 | 1/32, 2/2, 3/3, 4/4† | DSQ vs BWN, HWGQ, TWN, PACT, LQ-Net, ABC-Net, BCGD | 63.71, 65.17, 68.66, 69.56† | 69.90 | ❌ ImageNet |
| D6b | Table 7 | ImageNet, ResNet-34 | 2/2, 3/3, 4/4† | DSQ vs LQ-Net, ABC-Net, BCGD | 70.02, 72.54, 72.76† | 73.80 | ❌ ImageNet |
| D6c | Table 7 | ImageNet, MobileNetV2 | 4/4 | DSQ vs PACT (61.40) | 64.80 | 71.87 | ❌ ImageNet |
| D7 | Figure 3(b) | not stated | 2-bit | Accuracy vs α: DSQ, DSQ + sign, standard quantization | curves only | — | ✅ as analysis |
| D8 | Figure 5 | CIFAR-10, ResNet-20 | 2-bit | Weight distribution before and after DSQ | analysis | — | ✅ as analysis |
| D9 | Figure 6 | CIFAR-10, VGG-Small (binary); ImageNet, ResNet-34 (3-bit) | 1/1, 3/3 | Validation curves with and without DSQ | curves only | — | ❌ |
| D10 | Figure 7 | CIFAR-10, ResNet-20 | not stated | How α evolves during training | analysis | — | ✅ as analysis |
| D11 | Tables 1, 8, 9 | ARM deployment (Raspberry Pi 3B) | 2–4 bit | Inference speed: ResNet-18 at 2 bits takes 551.22 ms vs 935.51 ms for NCNN at 8 bits | speed only | — | ❌ not our topic |

**D3 in detail** (DSQ Table 4; "2-bit uniform quantization of ResNet-20 on CIFAR-10"):

| Variant | Top-1 | Top-5 | Gain over the previous row |
|---|---|---|---|
| Standard quantization | 86.63 | 99.35 | — |
| Fixed α | 86.95 | 99.50 | +0.32 |
| Learned α | 87.25 | 99.49 | +0.30 |
| Learned α, l, u | 88.44 | 99.50 | +1.19 |

Two-thirds of DSQ's total gain (+1.81) comes from learning the clipping bounds l and u. Our STE already learns them, EWGS-style.

DSQ uses two different full-precision numbers for ResNet-20: 90.84 in Table 3 and 90.78 in Table 6.

---

## Candidates for step 9

Only the ✅ and ⚠️ rows on CIFAR-10 ResNet-20 are realistic. Rough cost: one 400-epoch run takes about 2.1 h for the STE (measured per-epoch time) and about 2.9 h for EWGS or DSQ (estimated).

| Candidate | Runs | Cost | Why | Caveats |
|---|---|---|---|---|
| **E5c**: STE 84.7 vs EWGS 85.6, 1/1 | 2 | ~5 h | Same network, quantizer and protocol as EWGS's own code; the best possible check | One seed each, so the tolerance must cover seed noise |
| **D3**: DSQ, learned α, l, u = 88.44, 2-bit (+ our STE as context) | 2 | ~5 h | Our main bit width; DSQ's own ablation | W/A and schedule not stated; the network may use 1×1 downsampling convs; its "standard quantization" baseline does not learn l, u, so compare drops from full precision, not gains |
| E4: fixed δ = 1e-3 (85.3), 1/1 | 1 | ~3 h | Tests EWGS's backward rule without the Hessian: if E5c fails, this tells us whether the Hessian part is the problem | Optional; useful mainly for debugging |
| D2b / D5c: DSQ 1/32 = 90.24 | 1–2 | ~3–5 h | A second DSQ check, with float activations | Binarization details unclear |
| E5d / E5e: EWGS + DoReFa | 2–4 | ~5–10 h | — | Needs a DoReFa quantizer; E5e also quantizes the first and last layers |
| D2a / D5b: DSQ 1/1 = 84.11 | 1 | ~3 h | — | Different activation binarization (±1) and activation replacement |
| D4: PACT + DSQ = 90.11 | 2 | ~5 h | — | Needs PACT; model not stated |
