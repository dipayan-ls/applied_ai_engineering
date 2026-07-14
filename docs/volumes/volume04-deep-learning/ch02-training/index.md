# Ch 2 — Training Deep Networks

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain the mechanics of SGD, Momentum, RMSProp, and Adam, and select the appropriate optimiser for a given task.
2. Implement and compare learning-rate schedules including step decay, cosine annealing, linear warmup, and cyclic LR.
3. Apply batch normalisation and layer normalisation correctly and explain why each helps training.
4. Use dropout, L1/L2 regularisation, early stopping, and gradient clipping to control overfitting.
5. Write a production-quality PyTorch training loop with mixed-precision, gradient accumulation, and model checkpointing.

---

## 1. Optimisers

### 1.1 Stochastic Gradient Descent (SGD)

The simplest update rule. At step $t$, given gradient $g_t = \nabla_\theta \mathcal{L}$:

$$\theta_t = \theta_{t-1} - \eta \, g_t$$

**Vanilla SGD** is unbiased and memory-efficient but slow to converge — each noisy gradient step can point in an arbitrary direction.

### 1.2 SGD with Momentum

Exponentially weighted moving average of past gradients (the velocity $v_t$):

$$v_t = \mu v_{t-1} + g_t$$
$$\theta_t = \theta_{t-1} - \eta \, v_t$$

Typical $\mu = 0.9$. Momentum damps oscillations in high-curvature directions and accelerates movement in consistent directions, effectively increasing the step size along the loss valley.

**Nesterov Momentum** (look-ahead variant):

$$v_t = \mu v_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1} - \mu \eta v_{t-1})$$

### 1.3 RMSProp

Adapts the learning rate per-parameter by dividing by a moving average of squared gradients:

$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} \, g_t$$

Typical $\beta = 0.99$, $\epsilon = 10^{-8}$. Solves the problem of Adagrad's monotonically decreasing learning rate by using an exponential rather than cumulative average.

### 1.4 Adam (Adaptive Moment Estimation)

Adam combines Momentum (first moment) with RMSProp (second moment):

**First moment** (mean of gradients):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment** (uncentred variance of gradients):

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction** (initial moments are biased toward zero because $m_0 = v_0 = 0$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update**:

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3 \times 10^{-4}$.

!!! note "AdamW"
    The original Adam applies weight decay inside the adaptive term, which is incorrect. **AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update:
    $$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$
    AdamW is now the standard for Transformer training.

### 1.5 Optimiser Comparison

| Optimiser | Memory | Adaptive LR | Notes |
|-----------|--------|-------------|-------|
| SGD | $O(d)$ | No | Best final performance with tuning; used for CV |
| SGD + Momentum | $O(d)$ | No | Default for ResNet-style CV training |
| RMSProp | $O(d)$ | Per-param | Good for RNNs |
| Adam | $O(3d)$ | Per-param | Default for NLP / Transformers |
| AdamW | $O(3d)$ | Per-param | Preferred over Adam; correct weight decay |

---

## 2. Learning Rate Schedulers

A fixed learning rate is rarely optimal. Start too high and training diverges; keep it too high and you cannot settle into a sharp minimum.

### 2.1 Step Decay

Multiply $\eta$ by a factor $\gamma < 1$ every $k$ epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

Simple and interpretable. Common: $\gamma = 0.1$ every 30 epochs.

### 2.2 Cosine Annealing

Smoothly decreases the learning rate following a half-cosine from $\eta_{\max}$ to $\eta_{\min}$:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Avoids the abrupt drops of step decay. Widely used with restarts (SGDR).

### 2.3 Linear Warmup

Many Transformer training recipes warm up the learning rate linearly for $T_w$ steps:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_w} \quad \text{for} \quad t \le T_w$$

Warmup is critical for Adam because the first-moment estimates are unreliable early in training, and a high initial learning rate can cause instability.

### 2.4 Cyclic Learning Rate

Oscillates between $\eta_{\min}$ and $\eta_{\max}$ on a triangular wave. High learning rates act as simulated annealing — temporarily escaping local minima. Triangular schedule:

$$\eta_t = \eta_{\min} + \frac{(\eta_{\max} - \eta_{\min})}{2}\left(1 - \left|2 \cdot \text{frac}\!\left(\frac{t}{2T}\right) - 1\right|\right)$$

where $\text{frac}(x) = x - \lfloor x \rfloor$.

!!! tip "Practical Recommendation"
    For most Transformer fine-tuning: **linear warmup for 5% of total steps**, then **cosine decay to 0**. For CNNs on ImageNet: **step decay** (×0.1 at epochs 30, 60, 90) or cosine.

---

## 3. Batch Normalisation

### 3.1 Motivation

Internal covariate shift: the distribution of each layer's inputs changes as earlier layers' parameters update, forcing downstream layers to constantly readapt. Batch Normalisation (Ioffe & Szegedy, 2015) normalises each feature within a mini-batch.

### 3.2 Formula

For each feature $j$ over a mini-batch $\mathcal{B} = \{x_1, \ldots, x_N\}$:

$$\mu_\mathcal{B} = \frac{1}{N} \sum_{i=1}^{N} x_{ij}, \qquad \sigma^2_\mathcal{B} = \frac{1}{N} \sum_{i=1}^{N} (x_{ij} - \mu_\mathcal{B})^2$$

$$\hat{x}_{ij} = \frac{x_{ij} - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

$$y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j$$

where $\gamma_j$ (scale) and $\beta_j$ (shift) are **learnable parameters** that allow the network to undo the normalisation if beneficial.

### 3.3 Training vs Inference

During **training**: use mini-batch statistics $\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$.

During **inference**: use running (exponential) averages $\mu_{\text{run}}$, $\sigma^2_{\text{run}}$ accumulated during training. This makes the output deterministic and independent of batch size.

### 3.4 Why BatchNorm Helps

- Reduces sensitivity to weight initialisation.
- Acts as a form of regularisation (the normalisation noise is batch-dependent).
- Allows higher learning rates by smoothing the loss landscape.
- Speeds up convergence by roughly 2-10× in practice.

---

## 4. Layer Normalisation vs Batch Normalisation

### 4.1 Layer Normalisation

Normalises across the **feature dimension** for each example independently:

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma^2_i + \epsilon}}, \qquad \mu_i = \frac{1}{H}\sum_{j=1}^{H} x_{ij}$$

No batch dimension involved — statistics are computed per sample.

### 4.2 When to Use Each

| Criterion | Batch Norm | Layer Norm |
|-----------|-----------|-----------|
| Architecture | CNNs | Transformers, RNNs |
| Batch size dependency | Requires large batches (≥16) | Independent of batch size |
| Works at inference | Yes (running stats) | Yes (per-sample) |
| Sequence tasks | Problematic (variable length) | Natural |
| Image tasks | Very effective | Less common |

!!! warning "BatchNorm with Small Batches"
    With batch sizes < 8, batch statistics are too noisy. Use Group Norm or Layer Norm instead. This is a common source of bugs when porting models to inference with batch size 1.

---

## 5. Dropout

### 5.1 Mechanism

During training, each neuron is independently zeroed out with probability $p$ (typically $p = 0.1$–$0.5$):

$$\tilde{h}_j = \begin{cases} 0 & \text{with prob } p \\ h_j / (1-p) & \text{with prob } 1-p \end{cases}$$

The $1/(1-p)$ scaling is **inverted dropout** — it ensures the expected value of $\tilde{h}$ equals $h$, so no scaling change is needed at inference.

### 5.2 Training vs Inference

- **Training**: Apply dropout mask, scale by $1/(1-p)$.
- **Inference**: Use the full network (dropout disabled). In PyTorch, `model.eval()` automatically disables dropout.

### 5.3 Regularisation Effect

Dropout forces the network to not rely on any single neuron — each sub-network (defined by the dropout mask) must independently solve the task. This is equivalent to training an exponential ensemble of sub-networks.

---

## 6. L1 and L2 Regularisation

### 6.1 L2 Regularisation (Weight Decay)

Adds $\frac{\lambda}{2}\|W\|_F^2$ to the loss:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} \|W^{(l)}\|_F^2$$

Gradient: $\nabla_W \mathcal{L}_{\text{reg}} = \nabla_W \mathcal{L} + \lambda W$

This shrinks weights toward zero. In SGD update: $W \leftarrow (1 - \eta\lambda)W - \eta\nabla_W \mathcal{L}$.

### 6.2 L1 Regularisation (Sparsity)

Adds $\lambda \|W\|_1$ to the loss. The subgradient is $\lambda \operatorname{sign}(W)$, which pushes weights exactly to zero — producing sparse solutions. Rarely used in deep learning (not differentiable at zero).

### 6.3 In PyTorch

L2 via optimiser:
```python
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

L1 manually:
```python
l1_loss = sum(p.abs().sum() for p in model.parameters())
total_loss = ce_loss + lambda_l1 * l1_loss
```

---

## 7. Early Stopping and Checkpointing

**Early stopping** monitors a held-out validation metric. If the metric does not improve for `patience` epochs, training stops. This prevents overfitting without needing to specify the number of epochs upfront.

**Model checkpointing** saves the model state at each new validation best. The saved checkpoint is the final model — not the weights at the last epoch (which may have overfit).

```python
best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 10

for epoch in range(max_epochs):
    train(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
```

---

## 8. Mixed Precision Training

### 8.1 Why Mixed Precision?

Modern GPUs have specialised tensor cores that run FP16 operations at 2–8× the throughput of FP32. Mixed precision uses FP16 for the bulk of compute (forward pass, backward pass) while keeping master weights in FP32 for numerical stability.

### 8.2 FP16 vs BF16

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | High |
| FP16 | 16 | 5 | 10 | ±65504 | Medium |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | Low |

BF16 has the same dynamic range as FP32 (wide exponent) but lower mantissa precision. It is less prone to overflow than FP16 and is the recommended format for Transformer training on Ampere+ GPUs and TPUs.

### 8.3 Loss Scaling (for FP16)

FP16 cannot represent gradients smaller than $\sim 6 \times 10^{-8}$. Gradients in this range underflow to zero. **Loss scaling** multiplies the loss by a large constant $S$ before the backward pass, then divides the gradients by $S$ before the update:

$$\text{loss\_scaled} = \mathcal{L} \times S \implies \text{grad} \gets \frac{\text{grad}}{S}$$

Dynamic loss scaling (PyTorch `GradScaler`) automatically adjusts $S$ — increasing when no overflow is detected, decreasing otherwise.

---

## 9. Gradient Accumulation

When GPU memory limits batch size, gradient accumulation simulates a larger effective batch:

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{k} \sum_{j=1}^{k} \nabla_\theta \mathcal{L}_j$$

Accumulate gradients for $k$ micro-batches before calling `optimiser.step()` and `optimiser.zero_grad()`. This is mathematically equivalent to one step on the concatenated batch, assuming the loss is averaged (not summed) over the batch.

---

## 10. Full PyTorch Training Loop

```python
"""
Production-quality PyTorch training loop.

Features:
- DataLoader with prefetching
- AdamW optimiser with weight decay
- Cosine annealing scheduler with linear warmup
- Gradient clipping
- Mixed precision training (torch.amp)
- Gradient accumulation
- Model checkpointing with early stopping
- Validation loop
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimiser: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine decay schedule with linear warmup.

    Args:
        optimiser:           The optimiser to schedule.
        num_warmup_steps:    Number of linear warmup steps.
        num_training_steps:  Total number of training steps.
        eta_min_ratio:       Minimum LR as a fraction of peak LR.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Generic trainer for classification models.

    Args:
        model:              PyTorch model (nn.Module).
        train_loader:       DataLoader for training data.
        val_loader:         DataLoader for validation data.
        lr:                 Peak learning rate.
        weight_decay:       AdamW weight decay coefficient.
        max_epochs:         Maximum number of training epochs.
        grad_clip:          Max gradient norm for clipping (0 = disabled).
        accumulation_steps: Number of micro-batches to accumulate.
        patience:           Early stopping patience (epochs).
        checkpoint_path:    Path to save the best model checkpoint.
        use_amp:            Whether to use automatic mixed precision.
        device:             Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        patience: int = 10,
        checkpoint_path: str | Path = "best_model.pt",
        use_amp: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = use_amp and self.device.type == "cuda"

        self.model.to(self.device)

        # Optimiser: separate weight-decay and no-decay param groups
        decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]
        self.optimiser = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # Scheduler: warmup for 5% of total steps
        total_steps = len(train_loader) * max_epochs // accumulation_steps
        warmup_steps = max(1, total_steps // 20)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimiser, warmup_steps, total_steps
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # State
        self.best_val_loss = float("inf")
        self._patience_counter = 0

    # ------------------------------------------------------------------
    # One training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss over all batches."""
        self.model.train()
        total_loss = 0.0
        self.optimiser.zero_grad()

        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                # Scale by accumulation steps so effective LR doesn't change
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimiser)

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.scheduler.step()
                self.optimiser.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Evaluate on validation set. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for x_batch, y_batch in self.val_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(x_batch)
                loss = F.cross_entropy(logits, y_batch, reduction="sum")

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model state dict and training metadata."""
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_best(self) -> None:
        """Restore best checkpoint into the model."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model from epoch {ckpt['epoch']} "
              f"(val_loss={ckpt['val_loss']:.4f})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        """Train the model and return loss/accuracy history.

        Returns:
            Dictionary with keys 'train_loss', 'val_loss', 'val_acc'.
        """
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": []
        }

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            lr_now = self.optimiser.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{self.max_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}  "
                f"lr={lr_now:.2e}"
            )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                print("  --> Saved new best checkpoint.")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self.load_best()
        return history
```

---

## Exercises

1. **Adam convergence.** Starting from initial parameters $\theta_0 = 1.0$, $m_0 = v_0 = 0$, and using $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$, and a constant gradient $g = 0.5$ for 5 steps, compute $\theta_1$ through $\theta_5$ by hand. Verify that bias correction prevents premature large steps.

2. **Scheduler ablation.** Implement a training loop that trains a small MLP on MNIST and compares: no scheduler, step decay (×0.1 at epochs 5, 10), cosine annealing, and linear warmup + cosine. Plot validation accuracy vs epoch for each. Which converges fastest? Which reaches the lowest loss?

3. **BatchNorm position.** Research Pre-LN vs Post-LN for Transformers. Implement a 3-layer MLP with BatchNorm placed (a) before the activation and (b) after the activation. Train on CIFAR-10 and report whether position matters significantly.

4. **Gradient accumulation correctness.** Show mathematically that accumulating gradients over $k$ micro-batches with average loss is equivalent to computing the gradient on the full concatenated batch (hint: linearity of expectation). What breaks if you use sum loss instead of mean?

5. **Mixed precision debugging.** Deliberately create a scenario where FP16 gradients underflow to zero (e.g., very small learning rates and specific initialisations). Verify using `GradScaler` that loss scaling fixes the issue. Monitor the scaler's `get_scale()` value throughout training.

---

## Summary

| Component | Purpose | Key Hyperparameter |
|-----------|---------|-------------------|
| SGD + Momentum | Convex / vision training | $\mu = 0.9$, $\eta$ tuning |
| Adam / AdamW | Transformer / NLP | $\beta_1=0.9$, $\beta_2=0.999$, $\eta \sim 3\times10^{-4}$ |
| Cosine + Warmup | LR schedule | Warmup = 5% of steps |
| Batch Norm | Normalise per-feature per-batch | $\epsilon=10^{-5}$ |
| Layer Norm | Normalise per-token per-example | $\epsilon=10^{-6}$ |
| Dropout | Regularisation | $p = 0.1$–$0.3$ |
| L2 (weight decay) | Prevent large weights | $\lambda \sim 10^{-2}$–$10^{-4}$ |
| Early stopping | Prevent overfitting | Patience = 5–20 |
| Mixed precision | Speed and memory | BF16 preferred on Ampere+ |
| Gradient clipping | Stability | max\_norm $= 1.0$ |

Training deep networks is as much engineering as science. A bug in any one of these components — wrong batch norm statistics at inference, missing `zero_grad()`, loss not divided by accumulation steps, forgetting `model.eval()` — can silently produce poor results. The training loop above encapsulates these correctly; use it as a template and adapt as needed.
