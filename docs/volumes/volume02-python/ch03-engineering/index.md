---
title: "Ch 3 — Python Engineering Practices"
description: "Production-grade Python: project layout, pytest, mypy, logging, configuration, CLI tools, packaging, and CI/CD — the practices that separate a notebook experiment from a maintainable AI system."
---

# Ch 3 — Python Engineering Practices

!!! abstract "Chapter overview"
    Writing correct Python is necessary but not sufficient for production AI systems. Code must be
    testable, statically verifiable, configurable without source edits, deployable as a standalone
    package, and automatically validated on every pull request. This chapter covers the engineering
    layer that sits below the AI logic — the scaffolding that keeps a project healthy as it grows
    from a solo experiment to a team-maintained service.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Scaffold a Python AI project using the `src` layout and `pyproject.toml` such that the package
   is importable in development, installable as a wheel, and publishable to PyPI without any
   additional configuration.
2. Write a `pytest` test suite with fixtures, parametrize, and `unittest.mock` that achieves
   meaningful coverage of numerical AI code without depending on large models or datasets.
3. Enable `mypy --strict` on a codebase and resolve the most common error categories, including
   `Any` propagation, missing stubs, and unannotated return types.
4. Replace ad-hoc `print` statements with a structured logging pipeline (stdlib `logging` +
   `structlog`) that emits JSON in production and human-readable output in development.
5. Configure a GitHub Actions workflow that runs `ruff`, `mypy`, and `pytest` in parallel on every
   pull request and blocks merge on any failure.

---

## 3.1 Project Structure: The `src` Layout

### 3.1.1 Why `src`?

Placing your package under `src/` prevents the most common packaging footgun: accidentally importing
the local source directory instead of the installed package during tests, masking installation errors.

```
my_ai_project/
├── src/
│   └── my_ai/                  ← importable package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── transformer.py
│       └── utils/
│           ├── __init__.py
│           └── metrics.py
├── tests/
│   ├── conftest.py             ← shared fixtures
│   ├── test_loader.py
│   ├── test_transforms.py
│   └── test_metrics.py
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml
└── .env.example
```

### 3.1.2 `pyproject.toml` — Complete Example

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-ai"
version = "0.1.0"
description = "Applied AI library example"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "pydantic>=2.5",
    "structlog>=23.2",
    "typer>=0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "mypy>=1.7",
    "ruff>=0.1",
    "line-profiler>=4.1",
]

[project.scripts]
my-ai = "my_ai.cli:app"          # entry_point → CLI

[tool.hatch.build.targets.wheel]
packages = ["src/my_ai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q --cov=my_ai --cov-report=term-missing"

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore  = ["ANN101", "ANN102"]
```

### 3.1.3 `__init__.py` Conventions

```python
# src/my_ai/__init__.py
"""my_ai — Applied AI library."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("my-ai")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0+dev"

# Explicit public API — only what callers should import directly
from my_ai.models.transformer import TransformerModel
from my_ai.data.loader import DataLoader

__all__ = ["TransformerModel", "DataLoader", "__version__"]
```

Install in editable mode during development:

```bash
pip install -e ".[dev]"
```

---

## 3.2 Testing with pytest

### 3.2.1 Unit Tests — Basic Structure

```python
# tests/test_metrics.py
import numpy as np
import pytest

from my_ai.utils.metrics import cosine_similarity, top_k_accuracy


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_score_negative_one(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_raises(self) -> None:
        zero = np.zeros(3)
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity(zero, np.array([1.0, 2.0, 3.0]))
```

### 3.2.2 Fixtures

Fixtures are reusable setup helpers. Declare them in `conftest.py` to share across files.

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic RNG shared across the entire test session."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_embedding_table(rng: np.random.Generator) -> np.ndarray:
    """256-row, 64-dim embedding table — fast to create, large enough to test."""
    return rng.standard_normal((256, 64)).astype(np.float32)


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "the model converged after 10 epochs",
        "gradient clipping prevents exploding gradients",
        "attention is all you need",
    ]
```

```python
# tests/test_loader.py
import numpy as np

from my_ai.data.loader import embed_texts


def test_embed_texts_shape(
    small_embedding_table: np.ndarray,
    sample_texts: list[str],
) -> None:
    result = embed_texts(sample_texts, small_embedding_table)
    assert result.shape == (len(sample_texts), 64)
```

### 3.2.3 `parametrize` — Data-Driven Tests

```python
# tests/test_transforms.py
import numpy as np
import pytest

from my_ai.data.transforms import l2_normalise


@pytest.mark.parametrize(
    "shape",
    [(1, 8), (16, 256), (3, 1, 512)],
    ids=["1x8", "16x256", "3x1x512"],
)
def test_l2_normalise_unit_norm(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape).astype(np.float32)
    X_norm = l2_normalise(X, axis=-1)
    norms = np.linalg.norm(X_norm, axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize(
    "bad_input, error_type",
    [
        (np.array([]), ValueError),
        (np.zeros((4, 8)), ValueError),  # zero-norm rows
    ],
)
def test_l2_normalise_invalid_inputs(
    bad_input: np.ndarray, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        l2_normalise(bad_input, axis=-1)
```

### 3.2.4 Mocking with `unittest.mock`

```python
# tests/test_api_client.py
from unittest.mock import MagicMock, patch

import pytest

from my_ai.data.api_client import EmbeddingAPIClient


def test_client_retries_on_timeout() -> None:
    client = EmbeddingAPIClient(base_url="https://api.example.com", max_retries=3)

    call_count = 0

    def flaky_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("request timed out")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.status_code = 200
        return mock_resp

    with patch("requests.Session.post", side_effect=flaky_post):
        result = client.embed("hello world")

    assert call_count == 3
    assert len(result) == 3


def test_client_raises_after_max_retries() -> None:
    client = EmbeddingAPIClient(base_url="https://api.example.com", max_retries=2)

    with patch("requests.Session.post", side_effect=TimeoutError):
        with pytest.raises(TimeoutError):
            client.embed("hello")
```

---

## 3.3 Type Checking with mypy

### 3.3.1 Running mypy in Strict Mode

```bash
mypy --strict src/my_ai
```

`--strict` enables: `--disallow-untyped-defs`, `--disallow-any-generics`, `--warn-return-any`,
`--no-implicit-optional`, and several others. Resolve errors in this order:

1. Missing return type annotations.
2. Untyped function parameters.
3. `Any` propagation from third-party libraries (add stubs or `type: ignore`).
4. `Optional` misuse.

### 3.3.2 Common Error Patterns and Fixes

```python
from typing import Any
import numpy as np


# ERROR: Returning Any from typed function
# def get_embedding(model: Any, text: str):    ← mypy: missing return type
#     return model.encode(text)

# FIX: annotate the return
def get_embedding(model: Any, text: str) -> np.ndarray:
    result: np.ndarray = model.encode(text)
    return result


# ERROR: Item "None" of "Optional[str]" has no attribute "upper"
# def shout(s: str | None) -> str:
#     return s.upper()           ← mypy: value is not always str

# FIX: guard before use
def shout(s: str | None) -> str:
    if s is None:
        return ""
    return s.upper()


# ERROR: Dict[str, int] not compatible with Dict[str, Any]
# def process(cfg: dict[str, Any]) -> None: ...
# int_cfg: dict[str, int] = {"lr": 1}
# process(int_cfg)   ← mypy: dict is invariant in value type

# FIX: use Mapping (covariant) for read-only parameters
from collections.abc import Mapping


def process(cfg: Mapping[str, int]) -> None:
    lr = cfg["lr"]
    print(f"lr={lr}")
```

### 3.3.3 `Protocol` for Third-Party Code

```python
from typing import Protocol
import numpy as np


class TokenizerProtocol(Protocol):
    """Structural type that any tokenizer must satisfy."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...


def build_vocab_embeddings(
    tokenizer: TokenizerProtocol,
    dim: int,
) -> np.ndarray:
    """Works with any object satisfying TokenizerProtocol — no inheritance required."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((tokenizer.vocab_size, dim)).astype(np.float32)
```

### 3.3.4 Strategic `type: ignore` Comments

```python
import torch  # no bundled stubs — mypy complains without ignore

# Single-line suppression with error code (preferred over blanket ignore)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]

# Document WHY you are suppressing — future maintainers will thank you
model = torch.nn.Linear(128, 64)  # type: ignore[no-untyped-call]  # torch stubs incomplete
```

---

## 3.4 Logging

### 3.4.1 The `logging` Module vs `print`

| Concern | `print` | `logging` |
|---------|:-------:|:---------:|
| Severity levels | No | Yes (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| Runtime filtering | No | Yes (`setLevel`) |
| Structured output | Manual | Via formatters |
| Propagation to parent loggers | No | Yes (library-friendly) |
| Redirect to file / syslog / remote | Manual | Via handlers |

```python
# src/my_ai/utils/log.py
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# Usage in library code — never call logging.basicConfig() inside a library
log = get_logger(__name__)

def train_step(loss: float, step: int) -> None:
    log.info("step=%d loss=%.4f", step, loss)  # use % formatting — lazily evaluated
    if loss > 10.0:
        log.warning("Unusually high loss at step %d: %.4f", step, loss)
```

### 3.4.2 Structured Logging with `structlog`

`structlog` binds context to a logger and emits JSON in production — essential for log aggregation
systems (Datadog, Loki, CloudWatch).

Install: `pip install structlog`

```python
# src/my_ai/utils/structured_log.py
import logging
import sys

import structlog


def configure_logging(*, json_output: bool = False) -> None:
    """Call once at application startup — not inside library code."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)


# Usage
import structlog as _structlog

log = _structlog.get_logger(__name__)


def load_dataset(path: str, split: str) -> None:
    log = _structlog.get_logger(__name__).bind(split=split, path=path)
    log.info("loading_dataset")
    # ... load ...
    log.info("dataset_loaded", n_samples=10_000, dtype="float32")
```

---

## 3.5 Configuration Management

### 3.5.1 Pydantic Settings

Pydantic `BaseSettings` reads from environment variables and `.env` files, validates types, and
provides IDE-friendly access — far superior to `os.environ.get()` scattered throughout code.

Install: `pip install pydantic-settings python-dotenv`

```python
# src/my_ai/config.py
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MY_AI_",       # MY_AI_DATABASE_URL, MY_AI_API_KEY, etc.
        case_sensitive=False,
    )

    # Required — no default; raises ValidationError if missing
    api_key: str = Field(description="OpenAI-compatible API key")

    # Optional with defaults
    model_name: str = Field(default="gpt-4o-mini", description="Chat model identifier")
    embedding_model: str = Field(default="text-embedding-3-small")
    max_tokens: int = Field(default=2048, ge=1, le=32_768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    batch_size: int = Field(default=32, ge=1)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}; got {v!r}")
        return upper


# Singleton pattern — import this object, not the class
settings = Settings()  # type: ignore[call-arg]  # reads from env / .env file
```

`.env.example` (commit this; never commit `.env`):

```ini
MY_AI_API_KEY=sk-...
MY_AI_MODEL_NAME=gpt-4o-mini
MY_AI_MAX_TOKENS=2048
MY_AI_TEMPERATURE=0.7
MY_AI_BATCH_SIZE=32
MY_AI_LOG_LEVEL=INFO
```

---

## 3.6 CLI Interfaces

### 3.6.1 Comparison: argparse vs click vs typer

| Criterion | `argparse` | `click` | `typer` |
|-----------|:----------:|:-------:|:-------:|
| Stdlib (no install) | Yes | No | No |
| Type annotations | Weak | Decorator-based | Native (uses hints) |
| Autocompletion | Manual | Built-in | Built-in |
| Subcommands | Yes | Yes | Yes |
| Testing | Manual | `CliRunner` | `CliRunner` |
| Best for | Simple scripts; no deps | Large CLIs; decorator style | AI projects with Pydantic types |

### 3.6.2 `argparse`

```python
import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an embedding model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, help="Path to training data")
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Training on {args.device} for {args.epochs} epochs")


if __name__ == "__main__":
    main()
```

### 3.6.3 `click`

```python
import click


@click.group()
def cli() -> None:
    """my-ai command-line tools."""


@cli.command("train")
@click.option("--data-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--epochs", default=10, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--lr", default=1e-4, show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu")
def train_cmd(
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    """Fine-tune an embedding model."""
    click.echo(f"Training on {device} — {epochs} epochs")


@cli.command("evaluate")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--split", type=click.Choice(["val", "test"]), default="val")
def evaluate_cmd(checkpoint: str, split: str) -> None:
    """Evaluate a saved checkpoint."""
    click.echo(f"Evaluating {checkpoint} on {split} split")


if __name__ == "__main__":
    cli()
```

### 3.6.4 `typer` (Recommended for AI Projects)

```python
# src/my_ai/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="my-ai command-line tools", pretty_exceptions_enable=False)


@app.command("train")
def train(
    data_dir: Annotated[Path, typer.Option("--data-dir", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option(min=1, max=1000)] = 10,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    lr: Annotated[float, typer.Option(min=1e-7, max=1.0)] = 1e-4,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Fine-tune an embedding model."""
    typer.echo(f"Training on {device} for {epochs} epochs")
    output_dir.mkdir(parents=True, exist_ok=True)


@app.command("evaluate")
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(exists=True)],
    split: Annotated[str, typer.Option()] = "val",
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
) -> None:
    """Evaluate a saved checkpoint."""
    typer.echo(f"Evaluating {checkpoint} on {split}")


if __name__ == "__main__":
    app()
```

---

## 3.7 Packaging

### 3.7.1 Building a Wheel

```bash
pip install hatch          # or build: pip install build

# Build both sdist and wheel
hatch build
# or: python -m build

# Artifacts appear in dist/
ls dist/
# my_ai-0.1.0-py3-none-any.whl
# my_ai-0.1.0.tar.gz
```

### 3.7.2 Publishing to PyPI

```bash
pip install twine

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# After smoke-testing the TestPyPI install:
pip install --index-url https://test.pypi.org/simple/ my-ai

# Publish to production PyPI
twine upload dist/*
```

Store credentials in `~/.pypirc` or use API tokens via environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...          # API token from pypi.org
```

### 3.7.3 `entry_points` / `[project.scripts]`

The `[project.scripts]` table in `pyproject.toml` wires a Python function to a shell command:

```toml
[project.scripts]
my-ai = "my_ai.cli:app"          # package.module:callable
my-ai-train = "my_ai.cli:train"  # expose sub-commands individually
```

After `pip install my-ai`, users can run `my-ai train --data-dir ...` from any directory.

---

## 3.8 CI/CD with GitHub Actions

### 3.8.1 Complete Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true   # cancel in-progress runs on new push to same branch

jobs:
  lint:
    name: Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  typecheck:
    name: Type check (mypy)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: mypy --strict src/my_ai

  test:
    name: Test (pytest) — Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest --cov=my_ai --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: coverage.xml
          fail_ci_if_error: true

  build:
    name: Build wheel
    needs: [lint, typecheck, test]    # only build if all checks pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install hatch
      - run: hatch build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

### 3.8.2 Branch Protection Rules

Configure the following in **Settings → Branches → Branch protection rules** on GitHub:

```
Branch name pattern:  main
✔ Require a pull request before merging
✔ Require status checks to pass before merging
    Required checks: CI / Lint (ruff)
                     CI / Type check (mypy)
                     CI / Test — Python 3.11
                     CI / Test — Python 3.12
✔ Require branches to be up to date before merging
✔ Do not allow bypassing the above settings
```

### 3.8.3 Caching Dependencies

```yaml
# Optimise with pip cache (already shown above via setup-python cache: pip).
# For larger projects use a hash-based approach:
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

---

## Exercises

!!! exercise "Exercise 1 — Scaffold a project"
    Using the `src` layout and `pyproject.toml` described in section 3.1, create a new Python
    project called `embed_kit` that exposes:
    - A `cosine_similarity(a, b)` function in `embed_kit.metrics`.
    - A `l2_normalise(X)` function in `embed_kit.transforms`.
    - A `my-embed` CLI entry point with sub-commands `embed` and `eval`.

    Verify that `pip install -e ".[dev]"` and `my-embed --help` both succeed.

!!! exercise "Exercise 2 — pytest + mocking"
    Write a test suite for a function `fetch_embeddings(texts: list[str]) -> np.ndarray` that
    calls an external REST API. The suite must:
    1. Mock the HTTP call with `unittest.mock.patch`.
    2. Test the happy path (API returns valid embeddings).
    3. Test retry behaviour (API times out twice, then succeeds).
    4. Use `@pytest.mark.parametrize` to test three different batch sizes.

!!! exercise "Exercise 3 — mypy strict clean-up"
    Run `mypy --strict` on the `embed_kit` project from Exercise 1. Resolve **every** error
    without using `type: ignore`. Document the three errors that required the most thought.

!!! exercise "Exercise 4 — Structured logging"
    Add `structlog` to `embed_kit` so that:
    - In development (`MY_EMBED_ENV=dev`), logs are printed in coloured console format.
    - In production (`MY_EMBED_ENV=prod`), logs are emitted as newline-delimited JSON.
    - Every log line includes `request_id` bound at the start of each CLI invocation.

    Verify by running the CLI and inspecting the output with `| python -m json.tool`.

!!! exercise "Exercise 5 — Full CI pipeline"
    Add a `.github/workflows/ci.yml` to `embed_kit` that:
    - Runs `ruff check`, `mypy --strict`, and `pytest` on Python 3.11 and 3.12.
    - Uploads a coverage report to Codecov.
    - Builds a wheel only when all checks pass.
    - Uses `concurrency` to cancel in-progress runs on new pushes.

    Open a pull request on your fork and verify all jobs go green.

---

## Summary

| Practice | Key decision | Default choice |
|----------|-------------|---------------|
| Project structure | `src` or flat layout | `src` — prevents import confusion |
| Build backend | `hatchling`, `setuptools`, `flit` | `hatchling` — fastest, PEP 517/660 native |
| Testing | `pytest` | Fixtures in `conftest.py`; `parametrize` for data-driven cases |
| Mocking | `unittest.mock` vs `pytest-mock` | `unittest.mock` — zero extra deps |
| Type checking | `mypy` vs `pyright` | `mypy --strict` — widest CI adoption |
| `type: ignore` policy | Blanket vs error-code | Always include the error code: `# type: ignore[attr-defined]` |
| Logging | `print` vs `logging` vs `structlog` | `structlog` for services; stdlib `logging` for libraries |
| Configuration | `os.environ` vs `pydantic-settings` | `pydantic-settings` — validation + IDE completion |
| CLI | `argparse` vs `click` vs `typer` | `typer` for new AI projects — native type annotation support |
| CI | GitHub Actions | `lint → typecheck → test → build`; branch protection on all four |

Engineering practices compound: a project that starts with good structure, typed interfaces, and
automated checks costs far less to maintain than one that retrofits these disciplines later. The
patterns in this chapter are not overhead — they are the scaffolding that lets you move fast without
breaking things as your AI system evolves from experiment to production.
