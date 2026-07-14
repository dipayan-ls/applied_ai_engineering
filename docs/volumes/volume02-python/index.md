---
title: "Volume 2 — Python Engineering"
description: "Master Python as an engineering discipline — from the language data model to production-grade tooling — so every AI system you build rests on a dependable foundation."
---

# Volume 2 — Python Engineering

Python is the _lingua franca_ of applied AI, but fluency at the REPL is a long way from engineering excellence.
This volume treats Python as a precision instrument: we examine how the runtime actually works, how scientific
libraries exploit hardware-level memory layouts, and how professional teams structure, test, type-check, and
ship Python code at scale.
By the end you will write code that is not only correct today but maintainable by your future self and your
teammates for years to come.

---

## Chapter Dependency Graph

```mermaid
graph LR
    C1["Ch 1 · Python Fundamentals"] --> C2["Ch 2 · Scientific Python"]
    C1 --> C3["Ch 3 · Engineering Practices"]
    C2 --> C3

    style C1 fill:#1976D2,color:#fff,stroke:none
    style C2 fill:#388E3C,color:#fff,stroke:none
    style C3 fill:#7B1FA2,color:#fff,stroke:none
```

!!! info "Reading path"
    Chapters 1 and 2 can be read independently of each other if you already have a strong background in one
    area. Chapter 3 assumes you are comfortable with the material in both preceding chapters.

---

## Chapters at a Glance

| # | Chapter | What you will learn | Est. reading time |
|---|---------|---------------------|:-----------------:|
| 1 | [Python Fundamentals for AI](ch01-fundamentals/index.md) | Data model, core types, comprehensions, decorators, context managers, generators, dataclasses | 3 h |
| 2 | [Scientific Python](ch02-scientific-python/index.md) | NumPy internals, broadcasting, `einsum`, SciPy, advanced Pandas, Polars, profiling | 3 h 30 min |
| 3 | [Python Engineering Practices](ch03-engineering/index.md) | Project layout, pytest, mypy, logging, configuration, CLI tools, packaging, CI/CD | 3 h |

---

## Volume Learning Outcomes

After completing this volume you will be able to:

1. **Explain** Python's object model and predict the behaviour of reference-based assignments, mutable
   default arguments, and late-binding closures — the three most common sources of subtle AI-pipeline bugs.
2. **Write** NumPy code that exploits contiguous memory and broadcasting instead of Python loops, achieving
   10–100× speedups on array operations common in model pre-processing.
3. **Apply** `np.einsum` to express tensor contractions, attention-score computations, and batch matrix
   multiplications in a single readable expression.
4. **Structure** a Python AI project using the `src` layout, `pyproject.toml`, and a reproducible virtual
   environment so that onboarding a new contributor takes minutes rather than hours.
5. **Enforce** code quality automatically via `pytest`, `mypy --strict`, `ruff`, and a GitHub Actions
   workflow that gates every pull request.
