"""
ResNet-20 for CIFAR-10 (He et al. 2016), as used by both EWGS and DSQ.

Matches the official EWGS code (CIFAR10/custom_models.py):
  - option-A (parameter-free, zero-padding) shortcut, post-activation blocks;
  - widths 16/32/64 with 3 blocks each;
  - an extra BatchNorm1d(64) between global average pooling and the final linear layer;
  - Kaiming-normal initialization of conv and linear weights.

Quantized (later, in QAT): the 18 convolutions inside the blocks. The first conv, the final linear
layer and all BatchNorms stay in float. `conv_layer` lets QAT swap in a quantized conv for the
block convolutions without changing the parameter names, so a float checkpoint loads directly.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, conv_layer):
        super().__init__()
        self.conv1 = conv_layer(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # option A: subsample and zero-pad the channels, no parameters
        self.pad = (planes - in_planes) // 2 if (stride != 1 or in_planes != planes) else None

    def shortcut(self, x):
        if self.pad is None:
            return x
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes=10, conv_layer=nn.Conv2d):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)  # first layer: always float
        self.bn1 = nn.BatchNorm2d(16)
        blocks, in_planes = [], 16
        for planes, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(3):
                blocks.append(BasicBlock(in_planes, planes, stride if i == 0 else 1, conv_layer))
                in_planes = planes
        self.layers = nn.Sequential(*blocks)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)  # last layer: always float
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(self.bn2(out))
