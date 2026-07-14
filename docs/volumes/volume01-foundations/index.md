# Volume 1 — Foundations of Artificial Intelligence

This volume lays the conceptual and historical groundwork that every Applied AI Engineer must have before writing a single line of production code. We begin with what AI actually is — not the science-fiction caricature, but the engineering discipline — and trace the seventy-year arc from symbolic logic to billion-parameter foundation models. Along the way we build the mathematical intuition and Python tooling fluency that all subsequent volumes assume. By the end of Volume 1 you will be able to reason clearly about what AI systems can and cannot do, situate any current technique within the broader research lineage, and set up a reproducible development environment ready for hands-on experimentation.

---

## Chapter Dependency Map

The four chapters in this volume are designed to be read in order. Each chapter builds directly on the vocabulary and mental models introduced in the one before it.

```mermaid
flowchart LR
    CH0["Ch 0\nWelcome to Applied AI"]
    CH1["Ch 1\nHistory of AI"]
    CH2["Ch 2\nMathematics for AI"]
    CH3["Ch 3\nPython & Tools"]

    CH0 --> CH1
    CH1 --> CH2
    CH2 --> CH3

    style CH0 fill:#4f86c6,color:#fff,stroke:#2d5f99
    style CH1 fill:#4f86c6,color:#fff,stroke:#2d5f99
    style CH2 fill:#4f86c6,color:#fff,stroke:#2d5f99
    style CH3 fill:#4f86c6,color:#fff,stroke:#2d5f99
```

---

## Chapters at a Glance

| # | Chapter | Description | Est. Reading Time |
|---|---------|-------------|:-----------------:|
| 0 | [Welcome to Applied AI](ch00-welcome/index.md) | Defines AI and its sub-disciplines; introduces the Applied AI Engineer role and the modern AI stack | 45 min |
| 1 | [History of Artificial Intelligence](ch01-history/index.md) | Traces the full arc from the 1943 McCulloch-Pitts neuron through foundation models; explains the AI winters and deep learning renaissance | 60 min |
| 2 | [Mathematics for AI](ch02-mathematics/index.md) | Linear algebra, calculus, probability, and information theory through the lens of ML; every concept tied to a concrete algorithm | 90 min |
| 3 | [Python & the AI Toolchain](ch03-python-tools/index.md) | NumPy, pandas, Jupyter, virtual environments, and the modern ML library ecosystem; sets up the workspace used in every subsequent volume | 75 min |

---

## Prerequisites

!!! note "No Prior AI Knowledge Required"
    Volume 1 assumes **no previous exposure to artificial intelligence or machine learning**. You should be comfortable reading Python code at a beginner level and have passed a first course in college algebra. All mathematical notation is introduced from scratch in Chapter 2.

---

## Volume Learning Outcomes

After completing all four chapters in this volume, you will be able to:

- **Define** artificial intelligence, machine learning, deep learning, and generative AI precisely, and explain the relationship between these terms without conflating them.
- **Locate** any AI technique you encounter in research or industry within the historical lineage of the field, understanding what problems it was designed to solve and what trade-offs it inherits.
- **Explain** why the two AI winters occurred and why the deep learning renaissance of 2006–2012 proved durable in ways previous revivals did not.
- **Apply** core linear algebra operations (matrix multiplication, eigendecomposition, SVD) and probability concepts (Bayes' theorem, expectation, entropy) as they appear in ML algorithms.
- **Read** and critically evaluate an AI research paper, identifying claims, evidence, experimental setup, and limitations.
- **Configure** a reproducible Python development environment with conda or pyenv, and use NumPy, pandas, and Matplotlib fluently for data exploration and visualization.
- **Articulate** the role of the Applied AI Engineer — distinct from ML researcher and data scientist — including the responsibilities, skill set, and engineering trade-offs specific to deploying AI in production.
- **Recognise** common misconceptions about AI (sentience, infallibility, magic black boxes) and correct them with technically precise language.

---

!!! tip "How to Use This Volume"
    Each chapter is self-contained enough to revisit independently. The exercises at the end of every chapter are graded in difficulty: exercises 1–2 are recall, 3–4 are application, and 5 is open-ended research or design. Working through all five before moving on is strongly recommended — the later volumes assume fluency, not just recognition, of the concepts introduced here.
