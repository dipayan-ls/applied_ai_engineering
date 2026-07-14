---
title: "Ch 1 — Python Fundamentals for AI"
description: "A rigorous tour of the Python language features every AI engineer must command: the data model, type system, functional tools, decorators, context managers, iterators, and dataclasses."
---

# Ch 1 — Python Fundamentals for AI

!!! abstract "Chapter overview"
    This chapter is not an introduction to Python syntax — it is a systematic study of the language
    mechanisms that appear constantly in AI codebases but are most often misunderstood. We work from
    the inside out: first the object model, then built-in types, then the higher-order language features
    that make Python expressive enough to build entire ML frameworks.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe Python's object model in terms of identity, type, and value, and predict the outcome of
   reference-based assignment and aliasing.
2. Choose the correct built-in collection type for a given access pattern and explain mutability
   trade-offs in terms of hash-ability and thread safety.
3. Write list, dict, set, and generator comprehensions that are both readable and more efficient than
   their `for`-loop equivalents.
4. Implement decorators and context managers from scratch, using them to add cross-cutting concerns
   (timing, resource cleanup, retry logic) without modifying business logic.
5. Model AI pipeline components as dataclasses with full type annotations, understanding when to prefer
   frozen vs mutable instances and when to escalate to `attrs` or Pydantic.

---

## 1.1 The Python Data Model: Objects, Types, and References

Every value in Python is an **object** — a heap-allocated structure with three mandatory components:

| Component | CPython internal | Python introspection |
|-----------|-----------------|----------------------|
| **Identity** | Memory address of the object | `id(x)` |
| **Type** | Pointer to a `PyTypeObject` | `type(x)` |
| **Value** | Type-specific payload | varies |

Variable names are **references** (labels) that point to objects, not boxes that contain values.
This distinction drives many of Python's most surprising behaviours.

```python
# Two names, one object
a: list[int] = [1, 2, 3]
b: list[int] = a          # b is an alias, NOT a copy

b.append(4)
print(a)  # [1, 2, 3, 4]  ← a sees the mutation because a and b point to the same list
print(id(a) == id(b))  # True

# Assignment rebinds the name, not the object
b = [99]
print(a)  # [1, 2, 3, 4]  ← a is unaffected; b now points to a new list
```

!!! tip "Box-and-arrow mental model"
    Visualise variables as sticky notes and objects as boxes on a table.
    `a = [1, 2, 3]` puts a box on the table and sticks the "a" note on it.
    `b = a` sticks another note ("b") on the *same* box.
    `b = [99]` moves the "b" note to a *new* box — the old box keeps the "a" note.

### 1.1.1 The `is` vs `==` Distinction

```python
x: list[int] = [1, 2, 3]
y: list[int] = [1, 2, 3]

print(x == y)   # True  — same *value*
print(x is y)   # False — different *objects* (different id)

# CPython interns small integers and short strings as an optimisation.
# Never rely on `is` for value equality — use it only to test for None, True, False.
assert (x is None) == False  # correct idiom
assert (x == None) == False  # also correct but triggers linting warnings
```

### 1.1.2 Reference Counting and the GIL

CPython manages memory with reference counting: each object stores how many references point to it.
When that count reaches zero the object is immediately deallocated (with a cyclic garbage collector
as a fallback for reference cycles).

The **Global Interpreter Lock (GIL)** ensures that only one thread executes CPython bytecode at a
time, making reference-count updates thread-safe without per-object locks. The practical consequence:

- CPU-bound multi-threading in pure Python does **not** achieve true parallelism — use `multiprocessing`
  or a C extension that releases the GIL (NumPy, PyTorch).
- I/O-bound multi-threading works well because the GIL is released during I/O waits.

!!! note "Python 3.13+ free-threaded build"
    CPython 3.13 introduced an experimental build option (`--disable-gil`) that removes the GIL.
    Adoption in the scientific Python ecosystem is ongoing; check library release notes before targeting
    this build in production.

---

## 1.2 Core Types and Mutability

### 1.2.1 Mutability Reference Table

| Type | Mutable | Hashable | Typical AI use |
|------|:-------:|:--------:|----------------|
| `int` | No | Yes | Indices, counts, random seeds |
| `float` | No | Yes | Hyperparameters, loss values |
| `complex` | No | Yes | Signal processing (FFT) |
| `str` | No | Yes | Labels, token text, paths |
| `bytes` | No | Yes | Raw model weights, serialised data |
| `bytearray` | Yes | No | In-place binary manipulation |
| `tuple` | No | Yes (if elements are) | Immutable coordinate pairs, cache keys |
| `list` | Yes | No | Dynamic batches, mutable sequences |
| `dict` | Yes | No | Feature maps, config, tokeniser vocab |
| `set` | Yes | No | Deduplication, membership tests |
| `frozenset` | No | Yes | Immutable sets used as dict keys |

### 1.2.2 Integers and Floats

```python
# Python integers are arbitrary-precision — no overflow
huge: int = 2 ** 1000          # works without special libraries

# IEEE-754 double precision for float — the usual suspects
print(0.1 + 0.2 == 0.3)        # False
print(abs(0.1 + 0.2 - 0.3) < 1e-9)  # True — always use tolerance comparisons

import math
print(math.isfinite(float("inf")))  # False
print(math.isnan(float("nan")))     # True
```

### 1.2.3 Strings and Bytes

```python
text: str = "gradient descent"

# Strings are sequences of Unicode code points
print(len("café"))   # 4 — characters, not bytes
print("café".encode("utf-8"))  # b'caf\xc3\xa9' — 5 bytes (é is 2 bytes in UTF-8)

# f-strings (the modern way — prefer over .format() or %)
loss: float = 0.2413
epoch: int = 42
msg: str = f"Epoch {epoch:03d} | loss={loss:.4f}"
print(msg)  # Epoch 042 | loss=0.2413

# Raw bytes for binary data
weights: bytes = b"\x00\xff\x7f"
print(len(weights))  # 3
```

### 1.2.4 Lists, Tuples, Sets, and Dicts

```python
from typing import Any

# List — ordered, mutable, allows duplicates
batch: list[dict[str, Any]] = [
    {"input_ids": [101, 7592, 102], "label": 1},
    {"input_ids": [101, 3231, 102], "label": 0},
]

# Tuple — ordered, immutable; great for multi-return values
def split_dataset(
    data: list[Any], ratio: float = 0.8
) -> tuple[list[Any], list[Any]]:
    split = int(len(data) * ratio)
    return data[:split], data[split:]

train, val = split_dataset(batch)

# Set — unordered, unique elements; O(1) membership test
vocab: set[str] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]"}
print("[CLS]" in vocab)  # True — O(1)

# Dict — ordered by insertion (Python 3.7+), O(1) average get/set
token_to_id: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
print(token_to_id.get("[MASK]", -1))  # -1 — safe default
```

---

## 1.3 Comprehensions

### 1.3.1 List Comprehensions

```python
# Basic form: [expression for item in iterable if condition]
logits: list[float] = [2.1, -0.5, 3.8, 1.2, -1.0]

# Softmax numerators (no condition)
import math
exp_logits: list[float] = [math.exp(x) for x in logits]

# Filter: keep only positive logits
positive: list[float] = [x for x in logits if x > 0]

# Nested: flatten a batch of token lists
batch_tokens: list[list[int]] = [[101, 7592], [101, 3231, 102]]
flat: list[int] = [token for seq in batch_tokens for token in seq]
# → [101, 7592, 101, 3231, 102]
```

### 1.3.2 Dict Comprehensions

```python
# Invert a token→id mapping to id→token
token_to_id: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
id_to_token: dict[int, str] = {v: k for k, v in token_to_id.items()}

# Selective update: double the id of every token that starts with "["
adjusted: dict[str, int] = {
    k: (v * 2 if k.startswith("[") else v)
    for k, v in token_to_id.items()
}
```

### 1.3.3 Set Comprehensions

```python
sentences: list[str] = ["the cat sat", "the dog ran", "a cat ran"]

# Unique words across all sentences
unique_words: set[str] = {word for sentence in sentences for word in sentence.split()}
print(unique_words)  # {'the', 'cat', 'sat', 'dog', 'ran', 'a'}
```

### 1.3.4 Generator Expressions

A generator expression uses parentheses instead of brackets and produces values **lazily** — one at a
time, without building an in-memory list. Essential for streaming large datasets.

```python
import math

logits = [2.1, -0.5, 3.8, 1.2, -1.0]

# sum() consumes the generator without ever materialising the full list
total = sum(math.exp(x) for x in logits)

# Generators can be chained to form a memory-efficient pipeline
def load_lines(path: str):
    with open(path, encoding="utf-8") as fh:
        yield from fh  # each line is produced on demand

def tokenise(lines):
    for line in lines:
        yield line.strip().split()

# No line is read until next() is called somewhere downstream
pipeline = tokenise(load_lines("corpus.txt"))
```

### 1.3.5 When to Use Each Form

| Form | Use when |
|------|----------|
| List comprehension | You need a reusable, indexable result and the data fits in RAM |
| Dict comprehension | You are building a mapping from an iterable of pairs |
| Set comprehension | You need uniqueness and do not care about order |
| Generator expression | Data is large / streaming; you consume it only once |

---

## 1.4 Functions

### 1.4.1 Default Arguments, `*args`, and `**kwargs`

```python
from typing import Any


def train_epoch(
    model: Any,
    loader: Any,
    lr: float = 1e-3,          # default argument
    clip_grad: float | None = None,
) -> float:
    """Run one training epoch and return the mean loss."""
    total_loss = 0.0
    for batch in loader:
        loss = model.step(batch, lr=lr)
        total_loss += loss
    return total_loss / len(loader)


def log_metrics(*values: float, prefix: str = "train") -> None:
    """Accept any number of metric values; prefix is keyword-only."""
    for i, v in enumerate(values):
        print(f"{prefix}/metric_{i}: {v:.4f}")


def create_optimizer(params: Any, **kwargs: Any) -> dict[str, Any]:
    """Forward arbitrary keyword arguments to the underlying optimizer."""
    defaults = {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999)}
    return {**defaults, **kwargs, "params": params}
```

### 1.4.2 Keyword-Only Arguments

Any parameter after a bare `*` is keyword-only — it cannot be passed positionally.

```python
def evaluate(
    model: Any,
    loader: Any,
    *,                         # everything after here is keyword-only
    device: str = "cpu",
    amp: bool = False,
) -> dict[str, float]:
    ...
    return {"accuracy": 0.95, "loss": 0.12}


# evaluate(model, loader, "cuda")  ← TypeError: device must be keyword argument
result = evaluate(model, loader, device="cuda", amp=True)  # correct
```

---

## 1.5 Type Annotations

Python type annotations are checked statically by tools such as `mypy` or `pyright`, but are **not**
enforced at runtime (unless you use Pydantic or similar). They serve as machine-verifiable documentation.

```python
from __future__ import annotations  # postponed evaluation — allows forward references

from typing import Any, Optional, TypeVar, Union

T = TypeVar("T")  # generic type variable

# Basic annotations
def clip_value(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Optional[X] is equivalent to X | None (Python 3.10+ syntax)
def load_checkpoint(path: str, device: Optional[str] = None) -> dict[str, Any]:
    ...


# Union (pre-3.10 syntax — kept for library compatibility)
def parse_label(raw: Union[str, int]) -> int:
    return int(raw)


# Collection generics
def pad_sequences(
    sequences: list[list[int]],
    pad_id: int = 0,
    max_len: int | None = None,
) -> list[list[int]]:
    length = max_len or max(len(s) for s in sequences)
    return [s + [pad_id] * (length - len(s)) for s in sequences]


# Generic functions
def first(items: list[T]) -> T:
    if not items:
        raise IndexError("empty list")
    return items[0]
```

### 1.5.1 `Protocol` for Structural Subtyping

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


def embed_batch(encoder: Encoder, texts: list[str]) -> list[list[int]]:
    return [encoder.encode(t) for t in texts]


# Any class with matching encode/decode methods satisfies Encoder,
# without inheriting from it — duck typing with static verification.
```

---

## 1.6 Decorators

A decorator is a callable that takes a function and returns a (usually modified) function.
The `@decorator` syntax is pure syntactic sugar for `func = decorator(func)`.

### 1.6.1 How Decorators Work

```python
import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")  # captures the parameter signature of the wrapped function
R = TypeVar("R")    # return type


def timer(func: Callable[P, R]) -> Callable[P, R]:
    """Measure and print the wall-clock execution time of *func*."""

    @functools.wraps(func)  # preserves __name__, __doc__, etc.
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__qualname__} finished in {elapsed:.4f}s")
        return result

    return wrapper


@timer
def train_step(batch_size: int) -> float:
    time.sleep(0.05)  # simulate work
    return 0.312

loss = train_step(32)  # prints: train_step finished in 0.0502s
```

### 1.6.2 Decorator Factories (Decorators with Arguments)

```python
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def retry(max_attempts: int = 3, exceptions: tuple[type[Exception], ...] = (Exception,)):
    """Retry *func* up to *max_attempts* times on specified exception types."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed ({exc}); retrying...")
            raise RuntimeError("unreachable")  # satisfies type checker

        return wrapper

    return decorator


@retry(max_attempts=5, exceptions=(ConnectionError, TimeoutError))
def fetch_embeddings(url: str) -> list[float]:
    ...
```

### 1.6.3 Class-Based Decorators

```python
import functools
from collections.abc import Callable
from typing import Any


class RateLimit:
    """Allow at most *calls_per_second* calls to the decorated function."""

    def __init__(self, calls_per_second: float) -> None:
        self._interval = 1.0 / calls_per_second
        self._last_call: float = 0.0

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            elapsed = time.perf_counter() - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.perf_counter()
            return func(*args, **kwargs)

        return wrapper


@RateLimit(calls_per_second=5.0)
def call_llm_api(prompt: str) -> str:
    ...
```

---

## 1.7 Context Managers

The `with` statement guarantees that clean-up code runs even when an exception is raised — exactly
what you need for file handles, database connections, GPU memory arenas, and timed code blocks.

### 1.7.1 The Protocol: `__enter__` and `__exit__`

```python
class ManagedTempDir:
    """Create a temporary directory on enter; delete it on exit."""

    import pathlib
    import shutil
    import tempfile

    def __init__(self, prefix: str = "ai_run_") -> None:
        self._prefix = prefix
        self.path: "pathlib.Path | None" = None

    def __enter__(self) -> "pathlib.Path":
        import pathlib
        import tempfile
        self.path = pathlib.Path(tempfile.mkdtemp(prefix=self._prefix))
        return self.path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool:
        import shutil
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
        return False  # do not suppress exceptions


with ManagedTempDir() as tmp:
    (tmp / "features.npy").write_bytes(b"\x93NUMPY")
    print(tmp.exists())  # True
# Directory and all its contents are deleted here.
```

### 1.7.2 `contextlib.contextmanager` — The Generator Shortcut

```python
import contextlib
import time
from collections.abc import Generator


@contextlib.contextmanager
def timer_block(label: str) -> Generator[None, None, None]:
    """Print elapsed time for the code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] {elapsed:.4f}s")


with timer_block("data loading"):
    import time; time.sleep(0.1)   # simulate work
```

### 1.7.3 `contextlib.suppress`

```python
import contextlib
import pathlib

# Silently ignore FileNotFoundError — equivalent to try/except but more readable
with contextlib.suppress(FileNotFoundError):
    pathlib.Path("stale_cache.pkl").unlink()
```

---

## 1.8 Iterators and Generators

### 1.8.1 The Iterator Protocol

Any object that implements `__iter__()` (returning self) and `__next__()` (returning the next value or
raising `StopIteration`) is an iterator.

```python
from typing import Iterator


class BatchIterator:
    """Yields fixed-size batches from a flat list of samples."""

    def __init__(self, data: list[int], batch_size: int) -> None:
        self._data = data
        self._batch_size = batch_size
        self._pos = 0

    def __iter__(self) -> "BatchIterator":
        return self

    def __next__(self) -> list[int]:
        if self._pos >= len(self._data):
            raise StopIteration
        batch = self._data[self._pos : self._pos + self._batch_size]
        self._pos += self._batch_size
        return batch


for batch in BatchIterator(list(range(10)), batch_size=3):
    print(batch)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]
```

### 1.8.2 Generators: `yield` and Lazy Evaluation

A generator function contains at least one `yield` statement. Calling it returns a **generator object**
(which is itself an iterator) without executing any of the body — execution only proceeds when
`next()` is called.

```python
from collections.abc import Generator
from typing import Any


def infinite_sampler(
    data: list[Any], *, shuffle: bool = True
) -> Generator[Any, None, None]:
    """Yield samples forever, optionally shuffling each epoch."""
    import random
    indices = list(range(len(data)))
    while True:
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            yield data[i]


sampler = infinite_sampler([10, 20, 30, 40])
first_five = [next(sampler) for _ in range(5)]
```

### 1.8.3 `yield from` for Generator Composition

```python
from collections.abc import Generator


def chain_datasets(*datasets: list[int]) -> Generator[int, None, None]:
    for dataset in datasets:
        yield from dataset  # delegates to each inner iterable


for sample in chain_datasets([1, 2], [3, 4], [5]):
    print(sample, end=" ")
# 1 2 3 4 5
```

### 1.8.4 Two-Way Communication with `send()`

```python
from collections.abc import Generator


def running_mean() -> Generator[float, float, None]:
    """Send each new value; receive the updated mean."""
    total: float = 0.0
    count: int = 0
    while True:
        value: float = yield (total / count if count else 0.0)
        total += value
        count += 1


gen = running_mean()
next(gen)           # prime the generator
print(gen.send(10.0))  # 10.0
print(gen.send(20.0))  # 15.0
print(gen.send(30.0))  # 20.0
```

---

## 1.9 Dataclasses

### 1.9.1 Basic Usage

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    model_name: str
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 10
    tags: list[str] = field(default_factory=list)  # safe mutable default

    def lr_at_epoch(self, epoch: int) -> float:
        """Linear warm-up schedule."""
        warmup_steps = 1000
        return self.learning_rate * min(1.0, epoch / warmup_steps)


cfg = TrainingConfig(model_name="bert-base-uncased", batch_size=64)
print(cfg)
# TrainingConfig(model_name='bert-base-uncased', learning_rate=0.0001, ...)
```

### 1.9.2 Frozen Dataclasses

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    pad_id: int = 0
    unk_id: int = 1

    def is_special_id(self, token_id: int) -> bool:
        return token_id in (self.pad_id, self.unk_id)


cfg = TokenizerConfig(vocab_size=32_000)
# cfg.vocab_size = 50_000  ← FrozenInstanceError: cannot assign to field 'vocab_size'

# Frozen dataclasses ARE hashable — usable as dict keys and in sets
seen: set[TokenizerConfig] = {cfg}
```

### 1.9.3 `__post_init__` for Validation

```python
from dataclasses import dataclass


@dataclass
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0; got {total:.6f}")
        for name, ratio in [("train", self.train), ("val", self.val), ("test", self.test)]:
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"{name} ratio must be in (0, 1); got {ratio}")


good = SplitRatios(0.7, 0.15, 0.15)
# bad  = SplitRatios(0.7, 0.2, 0.2)  ← ValueError: Split ratios must sum to 1.0; got 1.1
```

---

## 1.10 Common Python Mistakes in AI Codebases

!!! danger "Mutable default arguments"
    ```python
    # WRONG — all calls share the SAME list object
    def add_feature(name: str, features: list[str] = []) -> list[str]:
        features.append(name)
        return features

    print(add_feature("loss"))    # ['loss']
    print(add_feature("acc"))     # ['loss', 'acc']  ← unexpected!

    # CORRECT — use None as sentinel
    def add_feature_safe(name: str, features: list[str] | None = None) -> list[str]:
        if features is None:
            features = []
        features.append(name)
        return features
    ```

!!! danger "Late-binding closures"
    ```python
    # WRONG — all lambdas capture the SAME variable i (its final value = 2)
    multipliers = [lambda x: x * i for i in range(3)]
    print([m(10) for m in multipliers])  # [20, 20, 20]

    # CORRECT — capture i's current value with a default argument
    multipliers = [lambda x, i=i: x * i for i in range(3)]
    print([m(10) for m in multipliers])  # [0, 10, 20]
    ```

!!! danger "Modifying a list while iterating over it"
    ```python
    samples = [1, -2, 3, -4, 5]

    # WRONG — skips elements
    for s in samples:
        if s < 0:
            samples.remove(s)

    # CORRECT — filter to a new list
    samples = [s for s in samples if s >= 0]
    ```

!!! danger "Using `assert` for input validation in production code"
    ```python
    # WRONG — assertions are disabled when Python runs with -O flag
    def normalise(x: list[float]) -> list[float]:
        assert len(x) > 0, "empty input"   # silently skipped with -O
        ...

    # CORRECT — raise explicitly
    def normalise(x: list[float]) -> list[float]:
        if not x:
            raise ValueError("normalise requires a non-empty list")
        ...
    ```

!!! danger "Catching `Exception` too broadly"
    ```python
    # WRONG — swallows KeyboardInterrupt, SystemExit, real bugs
    try:
        result = call_api()
    except Exception:
        result = default_value

    # CORRECT — be specific
    try:
        result = call_api()
    except (ConnectionError, TimeoutError) as exc:
        print(f"API unreachable: {exc}")
        result = default_value
    ```

---

## Exercises

!!! exercise "Exercise 1 — Reference tracing"
    Without running the code, predict the output of the following snippet.
    Then verify your prediction in a REPL and explain each line.
    ```python
    a = {"x": [1, 2, 3]}
    b = a.copy()
    b["x"].append(4)
    b["y"] = 99
    print(a)
    print(b)
    ```

!!! exercise "Exercise 2 — Generator pipeline"
    Write a generator pipeline that:
    1. Reads lines from a text file (one document per line).
    2. Tokenises each line by splitting on whitespace.
    3. Filters out tokens shorter than 3 characters.
    4. Yields individual tokens one at a time.

    The pipeline must never load the entire file into memory. Verify memory usage
    with `tracemalloc`.

!!! exercise "Exercise 3 — Decorator composition"
    Implement a `cache` decorator that stores the result of a pure function call keyed
    on its arguments (a simplified `functools.lru_cache`). Then compose it with the
    `timer` decorator from section 1.6.1. Verify that the second call to the same
    arguments is faster and does not execute the function body.

!!! exercise "Exercise 4 — Dataclass hierarchy"
    Model the following hierarchy as dataclasses with full type annotations:
    - `BaseModelConfig` with fields `hidden_size: int`, `num_layers: int`, `dropout: float`.
    - `TransformerConfig(BaseModelConfig)` adding `num_heads: int`, `ffn_size: int`.
    - Add `__post_init__` validation ensuring `hidden_size` is divisible by `num_heads`.
    - Make `BaseModelConfig` frozen.

!!! exercise "Exercise 5 — Type annotation audit"
    Take any Python script you have written previously and add complete `mypy --strict`-
    compliant type annotations. Run `mypy --strict` and resolve all errors. Document
    which errors surprised you the most.

---

## Summary

| Concept | Key takeaway |
|---------|-------------|
| Object model | Variables are references; assignment rebinds, not copies |
| Mutability | Immutable types (str, tuple, frozenset) are hashable; mutable are not |
| Comprehensions | Prefer list/dict/set comprehensions for clarity; generators for large data |
| Functions | Use `*` to enforce keyword-only args; `*args`/`**kwargs` for variadic interfaces |
| Type annotations | They are static metadata — enforce with mypy, not at runtime (unless Pydantic) |
| Decorators | `@functools.wraps` + `ParamSpec` preserves signatures through decoration layers |
| Context managers | `__exit__` always runs; `contextmanager` turns a generator into a context manager |
| Generators | Lazy, single-pass; use `yield from` to delegate; `send()` for two-way communication |
| Dataclasses | `frozen=True` for immutable value objects; `field(default_factory=...)` for mutable defaults |

Python's design rewards understanding the underlying mechanics. The patterns in this chapter appear
throughout NumPy, PyTorch, Hugging Face Transformers, and FastAPI — returning to this chapter as you
read later volumes will reveal familiar structures in new contexts.
