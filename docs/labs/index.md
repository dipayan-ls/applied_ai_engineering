# Labs

Hands-on labs accompany every volume. Each lab targets a specific skill and is graded by observable, reproducible output.

## Lab Structure

Every lab includes:

1. **Objective** — One sentence stating exactly what you will build
2. **Prerequisites** — Chapters that must be completed first
3. **Dataset** — Source, size, description
4. **Expected Output** — What a correct solution produces
5. **Starter Code** — Scaffold provided so you focus on the core concept
6. **Solution** — Full reference solution (attempt the lab first!)
7. **Extension Challenges** — 2-3 stretch goals for advanced students

---

## Lab Index

| Lab | Volume | Skill | Difficulty |
|---|---|---|---|
| Lab 01: NumPy from Scratch | Vol 1 | Scientific Python | Beginner |
| Lab 02: Linear Regression from Scratch | Vol 3 | ML Fundamentals | Beginner |
| Lab 03: Logistic Regression | Vol 3 | Supervised Learning | Intermediate |
| Lab 04: Decision Tree Implementation | Vol 3 | Supervised Learning | Intermediate |
| Lab 05: k-Means Clustering | Vol 3 | Unsupervised Learning | Intermediate |
| Lab 06: MLP in NumPy | Vol 4 | Neural Networks | Intermediate |
| Lab 07: PyTorch CNN — CIFAR-10 | Vol 4 | CNNs | Intermediate |
| Lab 08: Text Classification with LSTM | Vol 4 | RNNs | Intermediate |
| Lab 09: Attention from Scratch | Vol 5 | Transformers | Advanced |
| Lab 10: Fine-Tune BERT for Classification | Vol 5 | Pre-trained Models | Advanced |
| Lab 11: Build a RAG Pipeline | Vol 7 | RAG | Advanced |
| Lab 12: Agentic Search Tool | Vol 8 | AI Agents | Advanced |
| Lab 13: Deploy an LLM with FastAPI | Vol 9 | MLOps | Advanced |
| Lab 14: Monitor LLM Quality with Langfuse | Vol 9 | Monitoring | Advanced |

---

## Lab 01: NumPy from Scratch

**Objective:** Implement matrix multiplication, broadcasting, and a linear algebra solver using only NumPy — no SciPy, no sklearn.

**Prerequisites:** Vol 1 Ch 2 (Mathematics), Vol 1 Ch 3 (Python Tools)

**Dataset:** None (synthetic data)

**Tasks:**

1. Implement `matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray` without using `@` or `np.dot`
2. Implement `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
3. Implement PCA from scratch: centre, compute covariance matrix, eigendecompose, project
4. Verify your PCA matches `sklearn.decomposition.PCA` on the Iris dataset

```python
# Starter code
import numpy as np
from sklearn.datasets import load_iris

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply A @ B without using np.dot or @."""
    assert A.shape[1] == B.shape[0], "Incompatible shapes"
    # Your implementation here
    ...

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    ...

def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Project X onto its top n_components principal components."""
    ...
```

**Expected Output:**
```
matmul matches np.dot: True
cosine_similarity([1,0,0], [1,0,0]) = 1.0
cosine_similarity([1,0,0], [0,1,0]) = 0.0
PCA variance explained: [0.926, 0.053, 0.017, 0.005]
Custom PCA matches sklearn: True
```

---

## Lab 02: Linear Regression from Scratch

**Objective:** Implement linear regression using the normal equation and gradient descent; compare them on the California Housing dataset.

**Prerequisites:** Vol 1 Ch 2, Vol 3 Ch 1

```python
# Starter code
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegressionNormal:
    """Linear regression via the normal equation."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionNormal":
        ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

class LinearRegressionGD:
    """Linear regression via gradient descent."""
    def __init__(self, lr: float = 0.01, epochs: int = 1000) -> None:
        ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
```

**Extension:** Add Ridge regularisation (\(\lambda ||w||^2\)) to both implementations and plot the regularisation path.
