# Ch 1 — Neural Networks from Scratch

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain the biological inspiration for artificial neurons and formalise the perceptron model mathematically.
2. Compare activation functions — Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish — and justify when to use each.
3. Derive the forward-propagation equations for a multi-layer perceptron (MLP) and trace tensor shapes layer by layer.
4. Derive backpropagation from the chain rule and implement it without automatic differentiation.
5. Implement a fully functional 2-layer MLP in NumPy and reproduce it in PyTorch using `nn.Module`.

---

## 1. The Perceptron

### 1.1 Biological Inspiration

The neuron is the fundamental computational unit of the brain. A biological neuron receives electrochemical signals through **dendrites**, integrates them in the **cell body (soma)**, and — if the integrated signal exceeds a threshold — fires an **action potential** along the **axon** to downstream neurons through **synaptic junctions**.

Warren McCulloch and Walter Pitts (1943) proposed the first mathematical abstraction: a binary threshold unit. Frank Rosenblatt's **perceptron** (1958) added learnable weights and a simple update rule, making it the ancestor of every modern neural network.

### 1.2 Mathematical Model

A single neuron computes a weighted sum of its inputs and passes the result through a nonlinear activation function:

$$y = \sigma(w^T x + b)$$

where:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $x$ | $\mathbb{R}^n$ | Input feature vector |
| $w$ | $\mathbb{R}^n$ | Learnable weight vector |
| $b$ | $\mathbb{R}$ | Learnable bias (threshold shift) |
| $\sigma$ | $\mathbb{R} \to \mathbb{R}$ | Nonlinear activation function |
| $y$ | $\mathbb{R}$ | Neuron output |

!!! note "Why the Bias?"
    The bias $b$ shifts the activation function horizontally, allowing the neuron to fire even when all inputs are zero. Without a bias, every decision boundary would pass through the origin — a severe restriction.

---

## 2. Activation Functions

Without nonlinearity, composing linear transformations produces another linear transformation. Nonlinear activations are what give deep networks their expressive power.

### 2.1 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

Maps any real input to $(0, 1)$. Historically used in output layers for binary classification and in early hidden layers.

### 2.2 Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Derivative: $\tanh'(z) = 1 - \tanh^2(z)$

Maps inputs to $(-1, 1)$. Zero-centred, which helps gradient flow compared to Sigmoid.

### 2.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

Derivative: $\text{ReLU}'(z) = \mathbf{1}[z > 0]$ (1 if $z > 0$, else 0)

The workhorse of modern deep learning. Sparse activation, no saturation for positive inputs, computationally cheap.

### 2.4 Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}$$

Typically $\alpha = 0.01$. Solves the "dying ReLU" problem where neurons get stuck outputting zero permanently.

### 2.5 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$

where $\Phi$ is the cumulative distribution function of the standard normal. Approximation used in practice:

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)$$

Used in BERT, GPT-2, and most modern Transformers. Smooth, stochastic interpretation: gates inputs by how likely they are under a standard normal.

### 2.6 Swish

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

Discovered via neural architecture search (Ramachandran et al., 2017). Similar to GELU in practice, unbounded above, bounded below.

### 2.7 Comparison Table

| Activation | Range | Gradient Saturation | Zero-Centred | Common Uses |
|------------|-------|--------------------|--------------|----|
| Sigmoid | $(0, 1)$ | Yes (both sides) | No | Output (binary clf) |
| Tanh | $(-1, 1)$ | Yes (both sides) | Yes | RNN hidden states |
| ReLU | $[0, \infty)$ | Partial (negative side) | No | CNN, MLP hidden layers |
| Leaky ReLU | $(-\infty, \infty)$ | No | No | CNN when dying ReLU is a problem |
| GELU | $\approx(-0.17, \infty)$ | Rarely | Approximately | Transformers, BERT, GPT |
| Swish | $\approx(-0.28, \infty)$ | Rarely | Approximately | EfficientNet, modern CNNs |

!!! warning "The Dying ReLU Problem"
    If a ReLU neuron receives consistently negative pre-activations during training, its gradient is exactly zero. The neuron never updates and is permanently "dead." Solutions: Leaky ReLU, careful weight initialisation, smaller learning rates.

---

## 3. Multi-Layer Perceptron (MLP)

### 3.1 Architecture and Notation

An MLP stacks $L$ layers. We index layers $l = 1, 2, \ldots, L$ where $l=0$ is the input:

$$h^{(l)} = \sigma_l\!\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$$

| Symbol | Dimensions | Description |
|--------|-----------|-------------|
| $h^{(0)} = x$ | $\mathbb{R}^{d_0}$ | Network input |
| $W^{(l)}$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Weight matrix at layer $l$ |
| $b^{(l)}$ | $\mathbb{R}^{d_l}$ | Bias at layer $l$ |
| $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ | $\mathbb{R}^{d_l}$ | Pre-activation |
| $h^{(l)} = \sigma_l(z^{(l)})$ | $\mathbb{R}^{d_l}$ | Post-activation |
| $\hat{y} = h^{(L)}$ | $\mathbb{R}^{d_L}$ | Network output |

### 3.2 Batch Processing

In practice we process a mini-batch of $N$ examples simultaneously. All vectors become matrices:

$$Z^{(l)} = H^{(l-1)} W^{(l)T} + \mathbf{1} b^{(l)T} \in \mathbb{R}^{N \times d_l}$$

(or equivalently, with the transposed weight convention):

$$Z^{(l)} = H^{(l-1)} W^{(l)T} + b^{(l)}$$

where PyTorch uses `(N, d_in) @ (d_in, d_out) + (d_out,)` via broadcasting.

---

## 4. Forward Propagation — Concrete 2-Layer Example

Consider a network with:
- Input dimension $d_0 = 3$
- Hidden layer $d_1 = 4$ neurons, ReLU activation
- Output layer $d_2 = 2$ neurons, Softmax (for 2-class classification)
- Mini-batch $N = 5$

**Step 1 — Layer 1 pre-activation:**

$$Z^{(1)} = H^{(0)} W^{(1)T} + b^{(1)}$$

Shapes: $(5, 3) \times (3, 4) + (4,) = (5, 4)$

**Step 2 — Layer 1 post-activation (ReLU):**

$$H^{(1)} = \max(0, Z^{(1)}) \quad \in \mathbb{R}^{5 \times 4}$$

**Step 3 — Layer 2 pre-activation:**

$$Z^{(2)} = H^{(1)} W^{(2)T} + b^{(2)}$$

Shapes: $(5, 4) \times (4, 2) + (2,) = (5, 2)$

**Step 4 — Output (Softmax):**

$$\hat{Y}_{ij} = \frac{e^{Z^{(2)}_{ij}}}{\sum_{k} e^{Z^{(2)}_{ik}}} \quad \in \mathbb{R}^{5 \times 2}$$

Each row of $\hat{Y}$ sums to 1, giving a probability distribution over 2 classes.

---

## 5. Loss Functions

### 5.1 Mean Squared Error (Regression)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| y_i - \hat{y}_i \right\|^2$$

Penalises large errors quadratically. Gradient: $\nabla_{\hat{y}_i} \mathcal{L} = \frac{2}{N}(\hat{y}_i - y_i)$.

### 5.2 Cross-Entropy Loss (Classification)

For multi-class classification with $C$ classes, the true labels $y_i$ are one-hot vectors:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

**Derivation with Softmax output:**

Let logits be $z_c$ and softmax probabilities be $\hat{y}_c = e^{z_c} / \sum_k e^{z_k}$. The combined cross-entropy + softmax gradient is elegantly:

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_c} = \hat{y}_c - y_c$$

This is why the softmax + cross-entropy combination is almost always used together — the gradient has no exponential, making optimisation well-conditioned.

!!! tip "Numerical Stability of Softmax"
    To avoid overflow in $e^{z_c}$ when logits are large, subtract the maximum: $\hat{y}_c = e^{z_c - z_{\max}} / \sum_k e^{z_k - z_{\max}}$. This is mathematically equivalent but numerically stable.

---

## 6. Backpropagation

### 6.1 Intuition: The Chain Rule

Given a composite function $\mathcal{L} = f(g(h(x)))$, the chain rule gives:

$$\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

Backpropagation is the systematic application of the chain rule through the computational graph of a neural network, propagating error signals from the output back to every weight.

### 6.2 Full Derivation

Define the **error signal** at layer $l$ as:

$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \in \mathbb{R}^{d_l}$$

**Output layer** ($l = L$): For cross-entropy loss with softmax:

$$\delta^{(L)} = \hat{y} - y$$

**Hidden layers** (propagating backwards): Apply the chain rule through the activation:

$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'\!\left(z^{(l)}\right)$$

where $\odot$ denotes element-wise multiplication. This is the core backpropagation recursion.

**Weight gradients**: Once we have $\delta^{(l)}$, gradients for weights and biases at each layer:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(h^{(l-1)}\right)^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

In batch form ($N$ examples):

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{N} \left(\delta^{(l)}\right)^T H^{(l-1)}$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{N} \sum_{i=1}^{N} \delta^{(l)}_i$$

### 6.3 Weight Update

With learning rate $\eta$:

$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$

$$b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}$$

---

## 7. Vanishing and Exploding Gradients

### 7.1 Why They Happen

In a deep network, the error signal $\delta^{(l)}$ is the product of many Jacobian matrices stacked through the chain rule. At layer $l$ from the output:

$$\delta^{(l)} \propto \prod_{k=l}^{L-1} \left(W^{(k+1)}\right)^T \text{diag}\!\left(\sigma'\!\left(z^{(k)}\right)\right)$$

If the spectral norm of each Jacobian factor is less than 1 (e.g. Sigmoid derivatives max at 0.25), gradients shrink exponentially — **vanishing gradients**. If spectral norms exceed 1, gradients grow exponentially — **exploding gradients**.

### 7.2 Gradient Clipping

Clip the global gradient norm before each update:

$$\text{if} \quad \|\nabla\| > \tau: \quad \nabla \leftarrow \tau \cdot \frac{\nabla}{\|\nabla\|}$$

A common value is $\tau = 1.0$. In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

### 7.3 Weight Initialisation

Poor initialisation can cause vanishing/exploding gradients from the very first forward pass.

**Xavier / Glorot Initialisation** (designed for Sigmoid/Tanh):

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}},\ \sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}\right)$$

**He Initialisation** (designed for ReLU):

$$W \sim \mathcal{N}\!\left(0,\ \frac{2}{d_{\text{in}}}\right)$$

The factor of 2 compensates for ReLU setting half of activations to zero, preserving variance.

| Activation | Recommended Init | PyTorch Default |
|------------|-----------------|-----------------|
| Sigmoid, Tanh | Xavier uniform | `nn.init.xavier_uniform_` |
| ReLU | He normal | `nn.init.kaiming_normal_` |
| GELU, Swish | He normal (approximately) | `nn.init.kaiming_normal_` |

---

## 8. NumPy Implementation: 2-Layer MLP

```python
"""
Two-layer MLP implemented from scratch in NumPy.
Architecture: Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out) -> Softmax
Loss: Cross-entropy
Optimiser: Mini-batch SGD
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rectified linear unit: max(0, z)."""
    return np.maximum(0.0, z)


def relu_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise gradient of ReLU."""
    return (z > 0).astype(np.float64)


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable row-wise softmax.

    Args:
        z: Pre-softmax logits, shape (N, C).

    Returns:
        Probability matrix, shape (N, C). Each row sums to 1.
    """
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    y_hat: NDArray[np.float64],
    y: NDArray[np.int64],
) -> float:
    """Categorical cross-entropy loss.

    Args:
        y_hat: Predicted probabilities, shape (N, C).
        y:     Ground-truth class indices, shape (N,).

    Returns:
        Scalar mean loss.
    """
    n = y_hat.shape[0]
    log_probs = np.log(y_hat[np.arange(n), y] + 1e-12)
    return -float(log_probs.mean())


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP:
    """Two-layer MLP with ReLU hidden activation and softmax output.

    Args:
        d_in:     Input feature dimension.
        d_hidden: Number of hidden units.
        d_out:    Number of output classes.
        lr:       Learning rate for SGD.
        seed:     Random seed for reproducibility.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # He initialisation for ReLU-activated hidden layer
        self.W1: NDArray[np.float64] = rng.normal(
            0.0, np.sqrt(2.0 / d_in), size=(d_hidden, d_in)
        )
        self.b1: NDArray[np.float64] = np.zeros(d_hidden)

        # Xavier initialisation for output layer (no ReLU)
        self.W2: NDArray[np.float64] = rng.normal(
            0.0, np.sqrt(1.0 / d_hidden), size=(d_out, d_hidden)
        )
        self.b2: NDArray[np.float64] = np.zeros(d_out)

        self.lr = lr

        # Cache for backprop
        self._x: NDArray[np.float64] | None = None
        self._z1: NDArray[np.float64] | None = None
        self._h1: NDArray[np.float64] | None = None
        self._z2: NDArray[np.float64] | None = None
        self._y_hat: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run a forward pass.

        Args:
            x: Input batch, shape (N, d_in).

        Returns:
            Softmax probabilities, shape (N, d_out).
        """
        self._x = x                                  # (N, d_in)

        self._z1 = x @ self.W1.T + self.b1           # (N, d_hidden)
        self._h1 = relu(self._z1)                    # (N, d_hidden)

        self._z2 = self._h1 @ self.W2.T + self.b2   # (N, d_out)
        self._y_hat = softmax(self._z2)              # (N, d_out)

        return self._y_hat

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y: NDArray[np.int64]) -> None:
        """Compute gradients and update parameters.

        Args:
            y: Ground-truth class indices, shape (N,).
        """
        n = self._y_hat.shape[0]  # type: ignore[union-attr]

        # Output layer delta: softmax + cross-entropy combined gradient
        delta2 = self._y_hat.copy()                  # (N, d_out)
        delta2[np.arange(n), y] -= 1.0
        delta2 /= n

        # Gradients for W2, b2
        dW2 = delta2.T @ self._h1                    # (d_out, d_hidden)
        db2 = delta2.sum(axis=0)                     # (d_out,)

        # Propagate through hidden layer
        delta1 = delta2 @ self.W2                    # (N, d_hidden)
        delta1 *= relu_grad(self._z1)               # element-wise mask

        # Gradients for W1, b1
        dW1 = delta1.T @ self._x                    # (d_hidden, d_in)
        db1 = delta1.sum(axis=0)                    # (d_hidden,)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train the MLP using mini-batch SGD.

        Args:
            x_train:    Training inputs, shape (N, d_in).
            y_train:    Training labels, shape (N,).
            epochs:     Number of full passes over the data.
            batch_size: Mini-batch size.
            verbose:    Print loss every 10 epochs.

        Returns:
            List of per-epoch mean losses.
        """
        n = x_train.shape[0]
        history: list[float] = []
        rng = np.random.default_rng(0)

        for epoch in range(epochs):
            # Shuffle data
            perm = rng.permutation(n)
            x_shuf, y_shuf = x_train[perm], y_train[perm]

            epoch_losses: list[float] = []
            for start in range(0, n, batch_size):
                xb = x_shuf[start : start + batch_size]
                yb = y_shuf[start : start + batch_size]

                y_hat = self.forward(xb)
                loss = cross_entropy_loss(y_hat, yb)
                self.backward(yb)

                epoch_losses.append(loss)

            mean_loss = float(np.mean(epoch_losses))
            history.append(mean_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:4d}/{epochs}  loss={mean_loss:.4f}")

        return history

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return predicted class indices."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=1000, n_features=10, n_classes=3,
        n_informative=6, random_state=0
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MLP(d_in=10, d_hidden=64, d_out=3, lr=0.05)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32)

    preds = model.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"\nTest accuracy: {acc:.3f}")
```

---

## 9. PyTorch Implementation: 2-Layer MLP

```python
"""
Two-layer MLP implemented in PyTorch using nn.Module.
Equivalent architecture to the NumPy version above.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Two-layer MLP with ReLU hidden activation.

    Args:
        d_in:     Input feature dimension.
        d_hidden: Number of hidden units.
        d_out:    Number of output classes.
        dropout:  Dropout probability applied after hidden layer.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply He and Xavier initialisation to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        # Output layer: Xavier
        nn.init.xavier_uniform_(self.net[-1].weight)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits (pre-softmax).

        Args:
            x: Input batch, shape (N, d_in).

        Returns:
            Logits, shape (N, d_out).
        """
        return self.net(x)


def train_one_epoch(
    model: MLP,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimiser.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy) on a dataset."""
    model.eval()
    total_loss, correct = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += F.cross_entropy(logits, y_batch, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()

    n = len(loader.dataset)  # type: ignore[arg-type]
    return total_loss / n, correct / n


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # --- Data ---
    X, y = make_classification(
        n_samples=1000, n_features=10, n_classes=3,
        n_informative=6, random_state=0
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.tensor(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(X_te), torch.tensor(y_te))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    # --- Model ---
    model = MLP(d_in=10, d_hidden=64, d_out=3, dropout=0.1).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Training ---
    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimiser, device)
        if (epoch + 1) % 10 == 0:
            _, acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  test_acc={acc:.3f}")
```

---

## Exercises

1. **Activation gradient derivation.** Derive the derivative of the Swish function $f(z) = z \sigma(z)$ with respect to $z$. Show that it equals $\sigma(z) + z\sigma(z)(1 - \sigma(z))$.

2. **Shape tracing.** For an MLP with input dimension 784, hidden layers of sizes [512, 256, 128], and 10 output classes with a batch of 64, write out the shape of every weight matrix, bias vector, pre-activation matrix, and post-activation matrix.

3. **Backprop by hand.** Given a single neuron with $w = [0.5, -0.3]$, $b = 0.1$, input $x = [1.0, 2.0]$, and sigmoid activation, compute (a) the forward pass, (b) the loss gradient $d\mathcal{L}/dy = 1.0$, and (c) the gradients $d\mathcal{L}/dw$ and $d\mathcal{L}/db$.

4. **Initialisation experiment.** Modify the NumPy MLP to accept a choice of initialisation strategy (zeros, random uniform, Xavier, He). Train each variant on the `make_classification` dataset for 50 epochs and plot the loss curves. What do you observe?

5. **XOR problem.** The XOR function is not linearly separable. Train the NumPy MLP on the four XOR data points $\{(0,0) \to 0, (0,1) \to 1, (1,0) \to 1, (1,1) \to 0\}$. What is the minimum hidden layer size required for 100% training accuracy? Why?

---

## Summary

| Concept | Key Formula | Practical Note |
|---------|-------------|---------------|
| Perceptron | $y = \sigma(w^Tx + b)$ | Single computational unit |
| ReLU | $\max(0, z)$ | Default hidden activation |
| Forward pass | $h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$ | Cache $z^{(l)}$ for backprop |
| Cross-entropy | $-\frac{1}{N}\sum y \log \hat{y}$ | Combined with softmax |
| Backpropagation | $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$ | Core recursion |
| He init | $W \sim \mathcal{N}(0, 2/d_{\text{in}})$ | For ReLU activations |

The key insight of backpropagation is that it is not a special algorithm — it is the chain rule, applied systematically and cached efficiently. Modern automatic differentiation frameworks (PyTorch, JAX) do exactly this, but understanding the manual derivation is essential for debugging, designing custom architectures, and reasoning about gradient flow in novel network structures.
