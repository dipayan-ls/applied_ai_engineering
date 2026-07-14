---
title: "Ch 2 — Scientific Python"
description: "Deep-dive into NumPy memory layout, broadcasting, einsum, SciPy, advanced Pandas, Polars, and profiling — the computational toolkit every AI engineer must command."
---

# Ch 2 — Scientific Python

!!! abstract "Chapter overview"
    Scientific Python is not just a collection of libraries — it is a coherent philosophy: express
    computations as array operations rather than Python loops so that C and Fortran kernels can run
    at hardware speed. This chapter examines the internal machinery that makes this possible,
    gives you patterns for common AI pre-processing tasks, and teaches you to measure and eliminate
    bottlenecks before they reach production.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain how C-contiguous and Fortran-contiguous memory layouts affect the performance of NumPy
   slicing, transposition, and matrix multiplication, and choose layouts deliberately.
2. Write NumPy code that uses boolean masks and fancy indexing in place of Python-level loops for
   data selection and conditional updates.
3. Derive the output shape of any broadcasting operation by applying the three broadcasting rules,
   including operations across 3-D tensors.
4. Express attention score computation, batch matrix multiplication, and weighted sums using
   `np.einsum`, and explain why each notation is preferred over equivalent chained operations.
5. Profile a data-processing pipeline with `cProfile`, `line_profiler`, and `memory_profiler` to
   identify the exact lines responsible for performance regressions.

---

## 2.1 NumPy Internals: Memory Layout and Strides

### 2.1.1 The ndarray Data Model

A NumPy `ndarray` is a thin Python wrapper around a **contiguous block of typed memory** (the data
buffer). The array object stores metadata separately:

| Attribute | Meaning |
|-----------|---------|
| `ndarray.dtype` | Element type and byte width |
| `ndarray.shape` | Tuple of dimension sizes |
| `ndarray.strides` | Bytes to step in each dimension to reach the next element |
| `ndarray.data` | memoryview of the raw buffer |
| `ndarray.flags` | `C_CONTIGUOUS`, `F_CONTIGUOUS`, `WRITEABLE`, etc. |

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)

print(A.shape)    # (2, 3)
print(A.strides)  # (12, 4)  ← 12 bytes to move one row; 4 bytes to move one column
print(A.flags["C_CONTIGUOUS"])  # True  — row-major (C order)
```

### 2.1.2 C-Contiguous vs F-Contiguous

In **C order** (row-major) the last index varies fastest — moving along a row is a stride of one
element. In **Fortran order** (column-major) the first index varies fastest — moving along a column
is a stride of one element.

```python
C = np.ascontiguousarray(np.arange(12, dtype=np.float64).reshape(3, 4))
F = np.asfortranarray(C)

print("C strides:", C.strides)  # (32, 8) — 32 bytes / row; 8 bytes / col
print("F strides:", F.strides)  # (8, 24) — 8 bytes / row; 24 bytes / col
```

!!! warning "Transpose does not copy memory"
    ```python
    C = np.ones((1000, 1000), dtype=np.float32)
    T = C.T          # shape (1000, 1000) but strides are flipped — still F-contiguous
    print(T.flags["C_CONTIGUOUS"])  # False

    # Matrix multiply on a transposed view forces NumPy to handle non-contiguous strides.
    # For repeated use, materialise the contiguous layout:
    T_cont = np.ascontiguousarray(T)
    ```

### 2.1.3 Why Layout Matters for AI Workloads

BLAS (the C library behind `np.dot` and `@`) achieves peak throughput when both operands are
C-contiguous. A single inadvertent transpose in a hot loop can halve memory bandwidth utilisation.

```python
import time
import numpy as np

rng = np.random.default_rng(42)
A = rng.standard_normal((2048, 2048), dtype=np.float32)
B_T = rng.standard_normal((2048, 2048), dtype=np.float32).T  # non-contiguous

start = time.perf_counter()
for _ in range(10):
    _ = A @ B_T                          # NumPy handles internally
non_cont = time.perf_counter() - start

B_cont = np.ascontiguousarray(B_T)
start = time.perf_counter()
for _ in range(10):
    _ = A @ B_cont
cont = time.perf_counter() - start

print(f"Non-contiguous: {non_cont:.3f}s | Contiguous: {cont:.3f}s")
```

---

## 2.2 Advanced Indexing

### 2.2.1 Boolean (Mask) Indexing

```python
import numpy as np

logits = np.array([ 2.1, -0.5,  3.8,  1.2, -1.0,  0.4], dtype=np.float32)

# Select elements greater than zero
positive_mask = logits > 0.0           # array([True, False, True, True, False, True])
positive_logits = logits[positive_mask]  # array([2.1, 3.8, 1.2, 0.4])

# In-place conditional update — zero out negative logits
logits[logits < 0] = 0.0
print(logits)  # [2.1 0.  3.8 1.2 0.  0.4]

# 2-D example: mask rows where any feature is NaN
features = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
valid_rows = ~np.isnan(features).any(axis=1)  # [True, False, True]
clean_features = features[valid_rows]         # shape (2, 2)
```

### 2.2.2 Fancy (Integer Array) Indexing

```python
import numpy as np

embedding_table = np.random.randn(10_000, 256).astype(np.float32)  # vocab × dim

# Retrieve embeddings for a batch of token ids in one operation
token_ids = np.array([42, 1337, 7, 999, 42])        # shape (5,)
embeddings = embedding_table[token_ids]              # shape (5, 256)

# 2-D fancy index: select specific (row, col) pairs
scores = np.arange(20).reshape(4, 5)
rows = np.array([0, 1, 2, 3])
cols = np.array([4, 3, 2, 1])
diagonal_scores = scores[rows, cols]  # scores at positions (0,4), (1,3), (2,2), (3,1)
print(diagonal_scores)  # [ 4  8 12 16]
```

!!! tip "Fancy indexing always returns a copy"
    Boolean and fancy indexing return new arrays — mutations to the result do **not** propagate
    back to the source. Use `np.put` or direct boolean assignment (`arr[mask] = value`) for
    in-place updates.

---

## 2.3 Broadcasting

Broadcasting is NumPy's mechanism for applying element-wise operations to arrays of different shapes
without explicitly copying data.

### 2.3.1 The Three Broadcasting Rules

1. If arrays have different numbers of dimensions, prepend `1`s to the shape of the smaller array.
2. Arrays with size `1` along a dimension are **stretched** to match the other array's size in that
   dimension.
3. If the sizes disagree and neither is `1`, raise a `ValueError`.

### 2.3.2 2-D Example: Batch Normalisation

```python
import numpy as np

# batch: (32, 512) — 32 samples, 512 features
batch = np.random.randn(32, 512).astype(np.float32)

mean = batch.mean(axis=0)   # shape (512,)
std  = batch.std(axis=0)    # shape (512,)

# Broadcasting rules:
#   batch shape:  (32, 512)
#   mean shape:        (512,)  → prepend 1 → (1, 512)  → stretch → (32, 512)
normalised = (batch - mean) / (std + 1e-8)  # no copies made
print(normalised.shape)  # (32, 512)
```

### 2.3.3 3-D Example: Masked Self-Attention

Self-attention computes a scores matrix and applies a causal mask before softmax.

```python
import numpy as np

B, H, T, D = 2, 4, 8, 16   # batch, heads, seq_len, head_dim

# Query and Key: (B, H, T, D)
Q = np.random.randn(B, H, T, D).astype(np.float32)
K = np.random.randn(B, H, T, D).astype(np.float32)

# Attention scores: (B, H, T, T)
scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)

# Causal mask: (1, 1, T, T) — upper-triangular positions are -inf
mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
mask = mask[np.newaxis, np.newaxis, :, :]  # shape (1, 1, T, T)

# Broadcasting shapes:
#   scores shape: (B,  H,  T, T)  = (2, 4, 8, 8)
#   mask shape:   (1,  1,  T, T)  = (1, 1, 8, 8) → stretched to (2, 4, 8, 8)
masked_scores = scores + mask
print(masked_scores.shape)  # (2, 4, 8, 8)

# Softmax along last axis
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

attn_weights = softmax(masked_scores)
```

#### Shape derivation walkthrough

```
Operation          Left shape          Right shape        Output shape
─────────────────────────────────────────────────────────────────────
Q @ K.T            (2, 4, 8, 16)  →   (2, 4, 16, 8)  →  (2, 4, 8, 8)
scores + mask      (2, 4, 8, 8)       (1, 1, 8,  8)  →  (2, 4, 8, 8)
                                        ↑ stretched
```

---

## 2.4 `np.einsum`

`einsum` expresses tensor contractions using Einstein summation notation.
The string `'subscripts'` describes which indices are free (appear in output) and which are
contracted (appear only in inputs).

### 2.4.1 Syntax

```
np.einsum('ij,jk->ik', A, B)
          ─┬─ ─┬─  ─┬─
            │   │    └─ output index labels
            │   └─ index labels for second operand
            └─ index labels for first operand
```

Shared index `j` does not appear in the output → it is **summed over** (contracted).

### 2.4.2 Five Practical Examples

=== "Matrix multiply"
    ```python
    import numpy as np

    A = np.random.randn(64, 256).astype(np.float32)   # (batch, d_model)
    B = np.random.randn(256, 128).astype(np.float32)  # (d_model, d_out)

    # Equivalent to A @ B
    C = np.einsum("bi,io->bo", A, B)
    print(C.shape)  # (64, 128)
    ```

=== "Batch matrix multiply"
    ```python
    import numpy as np

    # Q, K: (batch, heads, seq_len, head_dim)
    Q = np.random.randn(2, 4, 8, 16).astype(np.float32)
    K = np.random.randn(2, 4, 8, 16).astype(np.float32)

    # Equivalent to Q @ K.transpose(0, 1, 3, 2)
    scores = np.einsum("bhid,bhjd->bhij", Q, K)
    print(scores.shape)  # (2, 4, 8, 8)
    ```

=== "Weighted sum (context vector)"
    ```python
    import numpy as np

    # attn_weights: (B, H, T, T); V: (B, H, T, D)
    attn_weights = np.random.dirichlet(np.ones(8), size=(2, 4, 8)).astype(np.float32)
    V = np.random.randn(2, 4, 8, 16).astype(np.float32)

    # Context vector: sum over key position j
    context = np.einsum("bhij,bhjd->bhid", attn_weights, V)
    print(context.shape)  # (2, 4, 8, 16)
    ```

=== "Outer product"
    ```python
    import numpy as np

    u = np.array([1.0, 2.0, 3.0])   # shape (3,)
    v = np.array([4.0, 5.0])        # shape (2,)

    # No shared index → outer product
    M = np.einsum("i,j->ij", u, v)
    print(M)
    # [[4. 5.]
    #  [8. 10.]
    #  [12. 15.]]
    ```

=== "Trace (sum of diagonal)"
    ```python
    import numpy as np

    A = np.arange(1, 10, dtype=np.float32).reshape(3, 3)

    trace = np.einsum("ii->", A)
    print(trace)  # 15.0  (1 + 5 + 9)
    # Equivalent to np.trace(A)
    ```

### 2.4.3 Performance: `optimize` Flag

```python
import numpy as np

# For chains of three or more operands, einsum can find the optimal contraction order
A = np.random.randn(100, 200).astype(np.float32)
B = np.random.randn(200, 300).astype(np.float32)
C = np.random.randn(300, 50).astype(np.float32)

# Let NumPy choose the cheapest contraction sequence
result = np.einsum("ij,jk,kl->il", A, B, C, optimize=True)
print(result.shape)  # (100, 50)
```

---

## 2.5 SciPy

### 2.5.1 Linear Algebra (`scipy.linalg`)

```python
import numpy as np
import scipy.linalg as la

A = np.array([[4.0, 3.0],
              [6.0, 3.0]])

# Determinant
print(la.det(A))         # -6.0

# Eigenvalues and right eigenvectors
eigenvalues, eigenvectors = la.eig(A)
print(eigenvalues.real)  # [9. -2.] (approximate)

# Solve Ax = b
b = np.array([10.0, 12.0])
x = la.solve(A, b)
print(x)                 # solution to the linear system

# SVD decomposition — common in PCA pre-processing
U, s, Vt = la.svd(A, full_matrices=False)
print(f"Singular values: {s}")
```

### 2.5.2 Optimisation (`scipy.optimize`)

```python
import numpy as np
from scipy.optimize import minimize


def negative_log_likelihood(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy for logistic regression."""
    w, b = params[:-1], params[-1]
    logits = X @ w + b
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return -float(np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


rng = np.random.default_rng(0)
X = rng.standard_normal((200, 5))
y = (X[:, 0] + rng.standard_normal(200) > 0).astype(np.float64)

init_params = np.zeros(X.shape[1] + 1)
result = minimize(
    negative_log_likelihood,
    x0=init_params,
    args=(X, y),
    method="L-BFGS-B",
    options={"maxiter": 500},
)
print(f"Converged: {result.success} | Loss: {result.fun:.4f}")
```

### 2.5.3 Statistics (`scipy.stats`)

```python
import numpy as np
from scipy import stats

a = np.array([2.1, 2.3, 2.05, 2.4, 2.15, 2.3])
b = np.array([1.9, 2.1, 2.0,  1.85, 2.0, 1.95])

# Two-sample t-test: do the two training runs have different mean loss?
t_stat, p_value = stats.ttest_ind(a, b)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

# Kolmogorov-Smirnov test: are two distributions the same?
ks_stat, ks_p = stats.ks_2samp(a, b)
print(f"KS={ks_stat:.3f}, p={ks_p:.4f}")

# Fit a normal distribution to data
loc, scale = stats.norm.fit(np.concatenate([a, b]))
print(f"Fitted μ={loc:.3f}, σ={scale:.3f}")
```

---

## 2.6 Advanced Pandas

### 2.6.1 MultiIndex

```python
import pandas as pd
import numpy as np

# Create a MultiIndex DataFrame: experiment × epoch
index = pd.MultiIndex.from_product(
    [["run_a", "run_b", "run_c"], range(1, 6)],
    names=["run", "epoch"],
)
rng = np.random.default_rng(42)
df = pd.DataFrame(
    {"train_loss": rng.uniform(0.1, 1.0, len(index)),
     "val_loss":   rng.uniform(0.15, 1.1, len(index))},
    index=index,
)

# Select all epochs for run_b
print(df.loc["run_b"])

# Cross-section: epoch 3 across all runs
print(df.xs(3, level="epoch"))

# Unstack: pivot runs to columns
pivot = df["val_loss"].unstack(level="run")
print(pivot)  # rows=epoch, cols=run
```

### 2.6.2 Time-Series Resampling

```python
import pandas as pd
import numpy as np

# Synthetic per-second latency log
rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=3600, freq="s")
latency = pd.Series(rng.exponential(scale=50.0, size=3600), index=idx, name="latency_ms")

# Downsample to 1-minute buckets
minute_stats = latency.resample("1min").agg(
    mean_latency=("latency_ms", "mean"),
    p95_latency=("latency_ms", lambda s: s.quantile(0.95)),
    count=("latency_ms", "count"),
)
print(minute_stats.head())

# Rolling 5-minute mean (window = 300 seconds)
rolling_mean = latency.rolling("5min").mean()
```

### 2.6.3 `apply` vs `transform` vs `agg`

| Method | Input | Output shape | Use for |
|--------|-------|:------------:|---------|
| `apply(func)` | Group DataFrame | Flexible | Arbitrary per-group computation |
| `transform(func)` | Group Series | Same as input | Broadcasting group statistics back to original rows |
| `agg(func)` | Group Series | One row per group | Summary statistics |

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "split": rng.choice(["train", "val", "test"], size=1000),
    "loss":  rng.exponential(0.5, size=1000),
})

# agg — one value per group
print(df.groupby("split")["loss"].agg(["mean", "std", "count"]))

# transform — broadcast group mean back to each row (for z-score normalisation)
df["loss_zscore"] = df.groupby("split")["loss"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-8)
)

# apply — return a new DataFrame per group
def top_k(group: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return group.nsmallest(k, "loss")

lowest_losses = df.groupby("split", group_keys=False).apply(top_k, k=3)
```

---

## 2.7 Memory Efficiency

### 2.7.1 Chunked Reading with Pandas

```python
import pandas as pd
from pathlib import Path


def stream_process_csv(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Compute per-category mean without loading the full file."""
    accumulator: dict[str, list[float]] = {}

    reader = pd.read_csv(path, chunksize=chunksize, dtype={"category": "category"})
    for chunk in reader:
        for cat, group in chunk.groupby("category", observed=True):
            accumulator.setdefault(str(cat), []).extend(group["value"].tolist())

    return pd.DataFrame(
        {cat: {"mean": sum(vals) / len(vals), "n": len(vals)}
         for cat, vals in accumulator.items()}
    ).T
```

### 2.7.2 Using Polars for Large Data

Polars uses Apache Arrow columnar memory and a Rust query engine — it is typically 5–20× faster than
Pandas on large aggregations and supports lazy evaluation.

```python
import polars as pl
from pathlib import Path


def analyse_large_log(path: Path) -> pl.DataFrame:
    """Lazy query: read, filter, group, aggregate — no data loaded until collect()."""
    return (
        pl.scan_csv(path, infer_schema_length=1000)
        .filter(pl.col("loss").is_not_nan())
        .filter(pl.col("split").is_in(["train", "val"]))
        .group_by(["split", "model_id"])
        .agg(
            pl.col("loss").mean().alias("mean_loss"),
            pl.col("loss").std().alias("std_loss"),
            pl.col("loss").min().alias("best_loss"),
            pl.len().alias("n_epochs"),
        )
        .sort(["split", "mean_loss"])
        .collect()  # triggers execution
    )
```

!!! tip "Choosing Pandas vs Polars"
    | Criterion | Prefer Pandas | Prefer Polars |
    |-----------|:-------------:|:-------------:|
    | Dataset size | < 1 GB in RAM | > 1 GB or streaming |
    | Ecosystem integration (sklearn, etc.) | Yes | Conversion needed |
    | Lazy / out-of-core execution | Limited | Native |
    | Multi-threaded aggregations | No (GIL) | Yes (Rust) |

---

## 2.8 Profiling

Measure before you optimise — the hotspot is rarely where you expect it.

### 2.8.1 `cProfile` — Function-Level

```python
import cProfile
import pstats
import io
import numpy as np


def preprocess(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-8)
    return np.clip(X, -5.0, 5.0)


def pipeline(n_samples: int = 100_000, n_features: int = 512) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    _ = preprocess(X)


prof = cProfile.Profile()
prof.enable()
pipeline()
prof.disable()

buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(15)
print(buf.getvalue())
```

Run from the command line:

```bash
python -m cProfile -s cumulative my_script.py | head -30
```

### 2.8.2 `line_profiler` — Line-Level

Install: `pip install line-profiler`

```python
# profile_demo.py
from line_profiler import profile  # use @profile decorator

@profile
def tokenise_batch(texts: list[str]) -> list[list[str]]:
    result = []
    for text in texts:
        tokens = text.lower().split()       # line A
        tokens = [t for t in tokens if t]   # line B — strip empty strings
        result.append(tokens)               # line C
    return result


if __name__ == "__main__":
    import random, string
    rng = random.Random(0)
    texts = [
        " ".join(
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
            for _ in range(rng.randint(5, 50))
        )
        for _ in range(50_000)
    ]
    tokenise_batch(texts)
```

```bash
kernprof -l -v profile_demo.py
```

### 2.8.3 `memory_profiler` — Line-Level Memory

Install: `pip install memory-profiler`

```python
# mem_demo.py
from memory_profiler import profile


@profile
def load_and_process(n: int = 500_000) -> None:
    import numpy as np
    X = np.random.randn(n, 128).astype(np.float32)   # line 1 — ~256 MB
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # line 2 — copy
    del X                                              # line 3 — freed
    _ = X_norm.astype(np.float16)                     # line 4 — half the memory


if __name__ == "__main__":
    load_and_process()
```

```bash
python -m memory_profiler mem_demo.py
```

### 2.8.4 `tracemalloc` — Built-in Memory Tracing

```python
import tracemalloc
import numpy as np

tracemalloc.start()

rng = np.random.default_rng(42)
X = rng.standard_normal((100_000, 256)).astype(np.float32)
y = X.sum(axis=1)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")[:5]
for stat in top_stats:
    print(stat)

tracemalloc.stop()
```

---

## Exercises

!!! exercise "Exercise 1 — Stride manipulation"
    Create a `(6, 6)` float32 array filled with values `0..35`. Using only `as_strided`
    (from `numpy.lib.stride_tricks`), extract a `(4, 4)` view that skips every other row
    and column without copying data. Verify with `np.shares_memory`.

!!! exercise "Exercise 2 — Broadcasting gradient accumulation"
    You have a weight matrix `W` of shape `(vocab_size, d_model)` and a batch of one-hot
    encoded token indices `X` of shape `(B, T, vocab_size)`. Using only broadcasting and
    `einsum` (no Python loops), compute the gradient `dL/dW` assuming `dL/dY` of shape
    `(B, T, d_model)` is given. Verify your result matches `(X.reshape(-1, vocab_size).T
    @ dL_dY.reshape(-1, d_model))`.

!!! exercise "Exercise 3 — einsum challenge"
    Implement scaled dot-product attention entirely using `np.einsum`:
    ```
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
    ```
    Inputs: `Q, K, V` each of shape `(B, H, T, D)`. Your solution must use exactly two
    `einsum` calls: one for scores and one for the context vector. Verify numerically
    against the explicit `@` implementation.

!!! exercise "Exercise 4 — Polars pipeline"
    Download the NYC Taxi trip dataset (public CSV, ~700 MB). Using a **lazy** Polars
    query, compute the median trip duration per pickup hour of day for weekdays only.
    Compare peak memory usage (via `tracemalloc`) against an equivalent Pandas pipeline.

!!! exercise "Exercise 5 — Profile and fix"
    The following function is slow:
    ```python
    def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        result = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                result[i, j] = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return result
    ```
    Profile it with `line_profiler`, then rewrite it using broadcasting and `einsum`
    to eliminate all Python loops. Benchmark with `A, B` of shape `(512, 256)` and
    report the speedup.

---

## Summary

| Topic | Key takeaway |
|-------|-------------|
| Memory layout | C-contiguous is the default; `np.ascontiguousarray` before repeated BLAS calls |
| Boolean indexing | Creates a copy; ideal for filtering; use direct assignment for in-place mutation |
| Fancy indexing | Also a copy; ideal for embedding table lookup and gather operations |
| Broadcasting | Apply 3 rules; prepend 1s; size-1 dimensions stretch; verify shapes explicitly |
| `einsum` | Prefer for multi-operand contractions; set `optimize=True` for three+ operands |
| SciPy | `linalg` for factorizations; `optimize.minimize` for any differentiable objective |
| Pandas advanced | MultiIndex for cross-sectional data; `transform` to broadcast group stats |
| Polars | Lazy API + Rust engine for > 1 GB data or when Pandas is the profiling hotspot |
| Profiling | Always profile before optimising; `cProfile` → `line_profiler` → `memory_profiler` |

The computational patterns in this chapter map directly to the pre-processing and post-processing
steps of every model in later volumes — you will recognise `einsum`-style notation in PyTorch's
`torch.einsum`, and broadcasting semantics in JAX's `jax.numpy`.
