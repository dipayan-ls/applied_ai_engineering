# Ch 3 — Convolutional Neural Networks

## Learning Objectives

By the end of this chapter you will be able to:

1. Formally define the discrete convolution operation and compute output dimensions given input size, kernel size, stride, and padding.
2. Explain feature maps, receptive fields, and why parameter sharing makes CNNs efficient for spatial data.
3. Describe the key innovations in LeNet, AlexNet, VGG, ResNet, and EfficientNet and trace the architectural lineage.
4. Implement a ResNet-style CNN in PyTorch from scratch, including skip connections and batch normalisation.
5. Apply transfer learning and data augmentation correctly, distinguishing feature extraction from fine-tuning.

---

## 1. The Convolution Operation

### 1.1 Formal Definition

For a 2D input feature map $I \in \mathbb{R}^{H \times W}$ and a kernel (filter) $K \in \mathbb{R}^{k_H \times k_W}$, the discrete cross-correlation (referred to as "convolution" in deep learning) is:

$$(I \star K)_{ij} = \sum_{m=0}^{k_H - 1} \sum_{n=0}^{k_W - 1} I_{(i \cdot S + m),\,(j \cdot S + n)} \cdot K_{mn}$$

where $S$ is the **stride**. True mathematical convolution flips the kernel, but the distinction vanishes when kernels are learned.

### 1.2 Hyperparameters

| Hyperparameter | Symbol | Effect |
|---------------|--------|--------|
| Kernel size | $K$ | Receptive field per layer |
| Stride | $S$ | Spatial downsampling |
| Padding | $P$ | Preserves spatial dimensions |
| Number of filters | $C_\text{out}$ | Depth of output feature map |

### 1.3 Output Size Formula

Given input spatial size $I$, kernel $K$, padding $P$, stride $S$:

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

**Examples:**

| Input | Kernel | Padding | Stride | Output |
|-------|--------|---------|--------|--------|
| $224$ | $3$ | $1$ | $1$ | $224$ (same) |
| $224$ | $3$ | $0$ | $1$ | $222$ (valid) |
| $224$ | $3$ | $1$ | $2$ | $112$ (halved) |
| $224$ | $7$ | $3$ | $2$ | $112$ (stem layer) |

### 1.4 Multi-Channel Convolution

For input $I \in \mathbb{R}^{C_\text{in} \times H \times W}$ and $C_\text{out}$ filters, each filter $K_c \in \mathbb{R}^{C_\text{in} \times k_H \times k_W}$:

$$(I \star K_c)_{ij} = \sum_{d=0}^{C_\text{in}-1} \sum_{m,n} I_{d,\, i \cdot S + m,\, j \cdot S + n} \cdot K_{c, d, m, n}$$

Output: $\mathbb{R}^{C_\text{out} \times O_H \times O_W}$.

Total parameters per conv layer (no bias): $C_\text{out} \times C_\text{in} \times k_H \times k_W$. Compare to a fully connected layer: $C_\text{out} \times (C_\text{in} \times H \times W)$ — the ratio $k_H k_W / (HW)$ shows the massive parameter saving.

!!! note "Why Parameter Sharing Works"
    A filter that detects a horizontal edge in the top-left of an image should detect the same edge anywhere. By sharing parameters across all spatial positions (translation equivariance), CNNs encode this inductive bias and require far fewer parameters than dense layers.

---

## 2. Feature Maps and Receptive Fields

The **feature map** is the activation volume produced by applying one filter across the entire spatial extent of the input. Each entry in the feature map corresponds to one **receptive field** — the region of the original input that influenced that entry.

The receptive field grows with depth. For a stack of $L$ layers each with kernel size $k$ and stride 1:

$$RF_L = 1 + L(k - 1)$$

With stride $S > 1$ or pooling, receptive fields grow faster. This is why deep networks with small kernels (VGG: all $3\times3$) can capture long-range spatial context.

**Effective receptive field**: In practice, not all pixels in the theoretical RF contribute equally — central pixels contribute exponentially more (due to the multiplicative paths). This motivates dilated convolutions (not covered here) for efficiently enlarging RFs.

---

## 3. Pooling Layers

Pooling reduces spatial dimensions, providing translation invariance and reducing computation.

### 3.1 Max Pooling

Takes the maximum value in each non-overlapping window:

$$\text{MaxPool}(I)_{ij} = \max_{m \in [0,k), n \in [0,k)} I_{ij \cdot S + m,\, ij \cdot S + n}$$

Preserves the strongest activation. Used in early CNNs and as downsampling.

### 3.2 Average Pooling

Takes the mean over each window. Smoother but less discriminative than max pooling for local features.

### 3.3 Global Average Pooling (GAP)

Averages an entire feature map to a single value per channel:

$$\text{GAP}(I)_c = \frac{1}{H \times W} \sum_{i,j} I_{c,i,j}$$

Produces a vector of size $C_\text{out}$ from a $C_\text{out} \times H \times W$ feature map. Replaces the final fully-connected layers in modern CNNs (ResNet, EfficientNet), drastically reducing parameters and enabling variable input sizes.

---

## 4. Standard Architectures

### 4.1 LeNet-5 (1998)

LeCun et al. applied to handwritten digit recognition (MNIST). Architecture: Conv → Pool → Conv → Pool → FC → FC → FC. Tanh activations. Demonstrated that convolutions learn useful features automatically.

### 4.2 AlexNet (2012)

Krizhevsky, Sutskever & Hinton. Won ImageNet LSVRC 2012 by a large margin (top-5 error: 15.3% vs 26.2% runner-up). Key innovations: ReLU activations, dropout, data augmentation, multi-GPU training. First modern deep CNN.

### 4.3 VGG (2014)

Simonyan & Zisserman. Showed that depth matters: replaced large kernels ($7\times7$, $11\times11$) with stacks of $3\times3$ convolutions. Two stacked $3\times3$ convolutions have the same receptive field as one $5\times5$ but fewer parameters and an extra ReLU.

### 4.4 ResNet (2015)

He et al. Key insight: residual (skip) connections enable training networks over 100 layers deep. Won ImageNet 2015.

### 4.5 EfficientNet (2019)

Tan & Le. Compound scaling: simultaneously scale depth, width, and resolution by a fixed ratio. Found via neural architecture search. State-of-the-art efficiency/accuracy trade-off for years.

### 4.6 Comparison Table

| Model | Params | ImageNet Top-1 | Top-5 | Year | Key Innovation |
|-------|--------|---------------|-------|------|---------------|
| AlexNet | 61 M | 56.5% | 80.3% | 2012 | Deep CNN + ReLU |
| VGG-16 | 138 M | 71.5% | 90.0% | 2014 | 3×3 conv stacks |
| ResNet-50 | 25 M | 76.2% | 92.9% | 2015 | Skip connections |
| ResNet-152 | 60 M | 78.3% | 94.1% | 2015 | Deeper ResNet |
| EfficientNet-B0 | 5.3 M | 77.1% | 93.3% | 2019 | Compound scaling |
| EfficientNet-B7 | 66 M | 84.3% | 97.0% | 2019 | Compound scaling |

---

## 5. Residual Connections

### 5.1 The Skip Connection

A residual block adds the block's input directly to its output:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

where $\mathcal{F}$ is typically two or three convolution layers with batch normalisation. If dimensions mismatch (different channels or stride), a $1\times1$ projection is used:

$$y = \mathcal{F}(x, \{W_i\}) + W_s x$$

### 5.2 Why Skip Connections Solve Vanishing Gradients

The gradient through a residual block is:

$$\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}(x)}{\partial x} + I$$

The identity term $I$ guarantees that the gradient never vanishes — even if $\partial \mathcal{F}/\partial x \approx 0$, the gradient is at least 1. In a network of $L$ residual blocks the gradient from the output to the input contains a sum over all $2^L$ subnetworks defined by different skip paths, preventing gradient from decaying exponentially.

### 5.3 Residual Block Variants

**Basic block** (ResNet-18, ResNet-34):
```
Input → Conv 3×3 → BN → ReLU → Conv 3×3 → BN → (+Input) → ReLU
```

**Bottleneck block** (ResNet-50, ResNet-101):
```
Input → Conv 1×1 → BN → ReLU
      → Conv 3×3 → BN → ReLU
      → Conv 1×1 → BN → (+Input) → ReLU
```

The bottleneck reduces the $3\times3$ convolution to operate on a narrower channel width, saving parameters.

---

## 6. Transfer Learning

Transfer learning leverages representations learned on large datasets (e.g. ImageNet, 1.28M images, 1000 classes) for new tasks.

### 6.1 Feature Extraction

Freeze all convolutional layers. Add a new classification head and train only the new layers:

```python
# Load pretrained backbone
backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False
# Replace head
in_features = backbone.fc.in_features
backbone.fc = nn.Linear(in_features, num_classes)  # Only this trains
```

Use when: small target dataset (< 1000 images per class), target domain similar to source.

### 6.2 Fine-Tuning

Unfreeze some or all backbone layers and train with a small learning rate:

```python
# Unfreeze last block only (layer4 in ResNet)
for param in backbone.layer4.parameters():
    param.requires_grad = True

# Different LR for head vs backbone
optimiser = torch.optim.AdamW([
    {"params": backbone.fc.parameters(), "lr": 1e-3},
    {"params": backbone.layer4.parameters(), "lr": 1e-4},
])
```

Use when: sufficient data (> 5000 samples), or when target domain differs from source.

| Strategy | When to Use | Training Time | Risk of Overfitting |
|---------|-------------|--------------|---------------------|
| Feature extraction | Small data, similar domain | Fast | Low |
| Partial fine-tuning | Medium data | Medium | Medium |
| Full fine-tuning | Large data or different domain | Slow | High |

---

## 7. Data Augmentation

Augmentation artificially increases dataset diversity, reducing overfitting.

### 7.1 Standard Augmentations

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),  # Random crop
    T.RandomHorizontalFlip(p=0.5),                # Mirror
    T.ColorJitter(brightness=0.4, contrast=0.4,   # Colour
                  saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),        # ImageNet stats
])
```

### 7.2 MixUp

Linearly interpolates two training examples and their labels:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha = 0.2$. Forces the network to behave linearly between training examples.

### 7.3 CutMix

Cuts a rectangular patch from one image and pastes it into another, mixing labels proportionally to the patch area:

$$\tilde{x} = M \odot x_i + (1 - M) \odot x_j$$

where $M$ is a binary mask with a rectangular cutout. Combines the regularisation benefit of Cutout (random erasing) with the mixed-label benefit of MixUp.

!!! tip "Augmentation Order Matters"
    Apply geometric transforms (crop, flip) before colour transforms. Always normalise last. Do not apply augmentations to the validation set — only normalisation.

---

## 8. PyTorch CNN Implementation: ResNet-Style

```python
"""
ResNet-style CNN with basic residual blocks, batch normalisation,
global average pooling, and transfer-learning-compatible design.

Supports:
- Variable number of stages and blocks
- Bottleneck or basic block
- Fine-tuning vs feature extraction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-layer residual block for ResNet-18/34.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        stride:       Stride for the first convolution (used for downsampling).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out = out + identity          # Skip connection: F(x) + x
        return F.relu(out, inplace=True)


class BottleneckBlock(nn.Module):
    """Three-layer bottleneck residual block for ResNet-50/101/152.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (expanded by factor 4).
        stride:       Stride for the 3×3 convolution.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        out = out + identity
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# Full ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """Configurable ResNet.

    Args:
        block:       Block class (BasicBlock or BottleneckBlock).
        layers:      Number of blocks per stage, e.g. [2, 2, 2, 2] for ResNet-18.
        num_classes: Number of output classes.
        in_channels: Number of input image channels (default 3 for RGB).
    """

    def __init__(
        self,
        block: type[BasicBlock | BottleneckBlock],
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self._in_channels = 64

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_stage(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_stage(block, 128, layers[1], stride=2)
        self.layer3 = self._make_stage(block, 256, layers[2], stride=2)
        self.layer4 = self._make_stage(block, 512, layers[3], stride=2)

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))          # Global average pool
        self.fc  = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_stage(
        self,
        block: type[BasicBlock | BottleneckBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage of residual blocks."""
        blocks = [block(self._in_channels, out_channels, stride)]
        self._in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """He initialisation for conv layers, zeros for BN biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits.

        Args:
            x: Input images, shape (N, C, H, W).

        Returns:
            Logits, shape (N, num_classes).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18: BasicBlock, [2, 2, 2, 2]."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50: BottleneckBlock, [3, 4, 6, 3]."""
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = resnet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"Output shape: {logits.shape}")   # (4, 10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")  # ~11M
```

---

## Exercises

1. **Output size derivation.** A CNN stem applies a $7\times7$ conv with stride 2, padding 3, then a $3\times3$ max pool with stride 2, padding 1. Starting from a $224\times224$ input, compute the output spatial dimensions after each operation. What is the effective downsampling factor?

2. **Parameter counting.** Compute the number of trainable parameters in a BasicBlock with $\text{in\_channels} = \text{out\_channels} = 64$ and no projection shortcut. Compare this to a fully connected layer mapping $64 \times 7 \times 7 = 3136$ inputs to 64 outputs.

3. **Receptive field.** In ResNet-18, each stage uses two $3\times3$ convolutions. Compute the cumulative receptive field size (in pixels of the original input) after each stage, accounting for strided downsampling. At which stage do neurons "see" the majority of the image?

4. **Transfer learning experiment.** Fine-tune a pretrained ResNet-18 (from `torchvision.models`) on a small dataset of your choice (e.g. CIFAR-10 resized to $224\times224$). Compare three settings: (a) train from scratch, (b) feature extraction (frozen backbone), (c) full fine-tuning. Report final test accuracy and training time for each.

5. **MixUp implementation.** Implement MixUp from scratch and add it to the CIFAR-10 training loop. Use $\alpha = 0.2$. Does it improve top-1 accuracy? How does it affect the calibration of predicted probabilities?

---

## Summary

| Concept | Formula / Key Idea | Practical Note |
|---------|-------------------|---------------|
| Conv output size | $O = \lfloor(I - K + 2P)/S\rfloor + 1$ | Check shapes before training |
| Parameter sharing | Same filter across all positions | Enables translation equivariance |
| Max pooling | $\max$ in $k\times k$ window | Use stride=2 for downsampling |
| Global avg pool | Mean per channel | Replaces FC; variable input size |
| Skip connection | $y = \mathcal{F}(x) + x$ | Enables 100+ layer networks |
| He init | $\mathcal{N}(0, 2/d_\text{in})$ | Essential with ReLU |
| Feature extraction | Freeze backbone, train head | Small data regime |
| Fine-tuning | Small LR on backbone | Large data or domain shift |
| MixUp | $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ | 0.5–1% accuracy gain |

CNNs dominated computer vision from 2012 to 2020. Though Vision Transformers have since surpassed them on large-scale benchmarks, ResNet remains the default backbone for transfer learning when data is limited, and the core ideas — convolutions, skip connections, GAP — appear as components inside modern architectures. Understanding CNNs deeply is a prerequisite for understanding what Transformers needed to do differently.
