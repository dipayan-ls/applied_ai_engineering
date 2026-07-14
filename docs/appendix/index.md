# Appendix

## Appendix A — Python Quick Reference

### Built-in Types Cheat Sheet

| Type | Mutable | Ordered | Example |
|---|---|---|---|
| `list` | Yes | Yes | `[1, 2, 3]` |
| `tuple` | No | Yes | `(1, 2, 3)` |
| `dict` | Yes | Yes (3.7+) | `{"a": 1}` |
| `set` | Yes | No | `{1, 2, 3}` |
| `frozenset` | No | No | `frozenset({1, 2})` |
| `str` | No | Yes | `"hello"` |
| `bytes` | No | Yes | `b"hello"` |

### NumPy Cheat Sheet

```python
import numpy as np

# Creation
a = np.array([1, 2, 3], dtype=np.float32)
z = np.zeros((3, 4))
o = np.ones((3, 4))
r = np.random.randn(3, 4)        # Standard normal
i = np.eye(3)                    # Identity

# Shape
a.shape          # (3,)
a.ndim           # 1
a.dtype          # float32
a.reshape(1, 3)  # (1, 3)
a.T              # Transpose

# Indexing
a[0]             # First element
a[-1]            # Last element
a[1:3]           # Slice
a[a > 0]         # Boolean mask
a[[0, 2]]        # Fancy indexing

# Operations
np.dot(a, b)     # Dot product (same as a @ b for 2D)
np.matmul(A, B)  # Matrix multiply (same as A @ B)
np.einsum('ij,jk->ik', A, B)  # Einstein summation
np.linalg.norm(a)  # L2 norm
np.linalg.eig(A)   # Eigendecomposition
np.linalg.svd(A)   # SVD
```

---

## Appendix B — Mathematics Reference

### Linear Algebra

| Identity | Formula |
|---|---|
| Dot product | \(a \cdot b = \sum_i a_i b_i = \|a\| \|b\| \cos\theta\) |
| Matrix multiply | \((AB)_{ij} = \sum_k A_{ik} B_{kj}\) |
| Transpose of product | \((AB)^T = B^T A^T\) |
| Inverse of product | \((AB)^{-1} = B^{-1} A^{-1}\) |
| Eigendecomposition | \(Av = \lambda v\) |
| SVD | \(A = U\Sigma V^T\) |
| Frobenius norm | \(\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}\) |

### Calculus

| Identity | Formula |
|---|---|
| Chain rule | \(\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)\) |
| Gradient | \(\nabla_x f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^T\) |
| Gradient descent | \(\theta \leftarrow \theta - \alpha \nabla_\theta L\) |
| Jacobian | \(J_{ij} = \partial f_i / \partial x_j\) |
| Hessian | \(H_{ij} = \partial^2 f / \partial x_i \partial x_j\) |

### Probability

| Identity | Formula |
|---|---|
| Bayes' Theorem | \(P(A|B) = P(B|A)P(A)/P(B)\) |
| Total probability | \(P(A) = \sum_b P(A|B=b)P(B=b)\) |
| Expectation | \(E[X] = \sum_x x P(X=x)\) |
| Variance | \(\text{Var}(X) = E[X^2] - (E[X])^2\) |
| Gaussian PDF | \(p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}\) |
| MLE | \(\hat\theta = \arg\max_\theta \log P(\mathcal{D}|\theta)\) |

### Information Theory

| Quantity | Formula |
|---|---|
| Entropy | \(H(X) = -\sum_x P(x)\log P(x)\) |
| Cross-entropy | \(H(P,Q) = -\sum_x P(x)\log Q(x)\) |
| KL Divergence | \(D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}\) |
| Mutual Information | \(I(X;Y) = H(X) - H(X|Y)\) |

---

## Appendix C — PyTorch Quick Reference

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Tensors
t = torch.tensor([1.0, 2.0, 3.0])
t = torch.zeros(3, 4)
t = torch.randn(3, 4)
t.shape          # torch.Size([3, 4])
t.dtype          # torch.float32
t.to("cuda")     # Move to GPU
t.cpu().numpy()  # → NumPy array

# Autograd
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()     # Compute gradients
x.grad           # Gradient tensor

# Neural Network
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MLP(784, 256, 10)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# Saving / loading
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

---

## Appendix D — Interview Question Bank

### Algorithms & Math

1. Prove that the softmax function is translation-invariant: \(\text{softmax}(z + c) = \text{softmax}(z)\). Why is this useful in practice?
2. Derive the gradient of cross-entropy loss with respect to the pre-softmax logits.
3. Explain why SVD is used for PCA instead of the full eigendecomposition of the data matrix.

### Deep Learning

4. Why do we use He initialisation for ReLU networks instead of Xavier?
5. What is the vanishing gradient problem? How does the LSTM architecture address it mathematically?
6. Compare batch norm and layer norm. In which architectures is each used and why?

### LLMs

7. Explain why the \(\sqrt{d_k}\) scaling in scaled dot-product attention is necessary.
8. What is the KV cache? How does its memory scale with sequence length and batch size?
9. Describe three techniques to reduce the memory footprint of LLM inference.

### System Design

10. Design a document ingestion and search system for a legal firm with 10 million PDFs. Address indexing latency, query latency, update frequency, and accuracy requirements.
11. Your LLM application is returning hallucinations 5% of the time in production. Walk through your investigation and remediation plan.
12. Design a multi-tenant LLM API platform that serves 500 enterprise customers with strict data isolation and cost attribution.
