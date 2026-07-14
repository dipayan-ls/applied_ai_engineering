---
title: "Ch 1 — ML Infrastructure"
---

# Ch 1 — ML Infrastructure

!!! info "Chapter Meta"
    **Level:** Advanced &nbsp;|&nbsp; **Reading time:** 90 min &nbsp;|&nbsp; **Volume:** 9 — MLOps

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Set up an MLflow tracking server and log metrics, parameters, and artefacts from a training run with full reproducibility.
2. Configure Weights & Biases for experiment tracking and run a Bayesian hyperparameter sweep across multiple parameters.
3. Design a feature store architecture that serves consistent, point-in-time-correct features for both offline training and online inference.
4. Version datasets and define reproducible ML pipelines as DAGs using DVC.
5. Build multi-stage GPU-enabled Docker images for ML workloads and deploy them to Kubernetes with resource limits and autoscaling.

---

## The ML Lifecycle

Every production ML system moves through a repeating lifecycle. Infrastructure choices at each stage determine cost, reproducibility, and team velocity. The loop is intentional: monitoring signals feed back into data collection and feature engineering.

```mermaid
flowchart LR
    DC["Data\nCollection"] --> FE["Feature\nEngineering"]
    FE --> TR["Model\nTraining"]
    TR --> EV["Evaluation"]
    EV --> DP["Deployment"]
    DP --> MO["Monitoring"]
    MO -->|"drift / degradation"| DC
```

Infrastructure that does not support this loop — where monitoring is bolted on afterward, or features are computed differently during training and serving — creates the *training-serving skew* that silently degrades production models over time.

---

## Experiment Tracking

Experiment tracking records the inputs (hyperparameters, data versions, code commits) and outputs (metrics, artefacts, plots) of every training run. Without it, a team cannot answer: "Which configuration produced our best model six weeks ago?"

### MLflow

MLflow is an open-source platform covering the complete ML lifecycle: Tracking, Projects, Models, and Registry.

**Server setup:**

```bash
pip install mlflow psycopg2-binary boto3

# Development: SQLite backend + local artefact store
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Production: PostgreSQL + S3
mlflow server \
  --backend-store-uri postgresql://user:pass@db-host:5432/mlflow \
  --default-artifact-root s3://my-ml-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Full training run with parameter, metric, and artefact logging:**

```python
"""
mlflow_training.py — MLflow experiment tracking example.

Requirements:
    pip install mlflow scikit-learn pandas matplotlib
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_track(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    experiment_name: str = "breast-cancer-gbt",
) -> str:
    """
    Train a Gradient Boosting classifier with full MLflow tracking.

    Returns:
        The MLflow run ID for downstream reference.
    """
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(experiment_name)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    with mlflow.start_run() as run:
        # ── Log hyperparameters ──────────────────────────────────────────────
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "dataset": "sklearn:breast_cancer",
                "n_features": data.data.shape[1],
                "n_train_samples": len(X_train),
            }
        )

        # ── Train ────────────────────────────────────────────────────────────
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # ── Log artefacts ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        importances = pd.Series(
            model.feature_importances_, index=data.feature_names
        )
        importances.sort_values().tail(10).plot.barh(ax=ax)
        ax.set_title("Top 10 Feature Importances")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importances.png")
        plt.close(fig)

        # ── Register model ───────────────────────────────────────────────────
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:3],
            registered_model_name="breast-cancer-gbt",
        )

        print(f"Run ID: {run.info.run_id}  |  Metrics: {metrics}")
        return run.info.run_id


def load_production_model(model_name: str = "breast-cancer-gbt") -> object:
    """Load the model currently in the Production stage of the registry."""
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


if __name__ == "__main__":
    for lr in [0.05, 0.1, 0.2]:
        for depth in [3, 5, 7]:
            train_and_track(n_estimators=200, max_depth=depth, learning_rate=lr)
```

---

### Weights & Biases

W&B provides a cloud-native experiment tracking platform with richer visualisations, native sweep support, and strong collaboration features.

**Initialising a run and logging training metrics:**

```python
"""
wandb_training.py — W&B experiment tracking with hyperparameter sweeps.

Requirements:
    pip install wandb torch transformers datasets
"""
from __future__ import annotations

import wandb


def train_one_run(config: dict | None = None) -> None:
    """Train a single model run, optionally driven by a W&B sweep."""
    with wandb.init(config=config) as run:
        cfg = wandb.config

        # Build model with sweep-provided hyperparameters
        model = build_transformer_model(
            model_name=cfg.model_name,
            dropout=cfg.dropout,
        )
        optimizer = build_optimizer(model, lr=cfg.learning_rate, wd=cfg.weight_decay)
        train_loader, val_loader = get_dataloaders(batch_size=cfg.batch_size)

        for epoch in range(cfg.epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                loss, acc = train_step(model, optimizer, batch)
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/accuracy": acc,
                        "epoch": epoch,
                        "global_step": epoch * len(train_loader) + step,
                    }
                )

            val_metrics = evaluate(model, val_loader)
            wandb.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"],
                    "epoch": epoch,
                }
            )

        # Save best checkpoint as a W&B artefact
        artifact = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            metadata={"val_accuracy": val_metrics["accuracy"]},
        )
        artifact.add_file("checkpoint_best.pt")
        run.log_artifact(artifact)
```

**Hyperparameter sweep with Bayesian optimisation:**

```python
import wandb

sweep_config: dict = {
    "method": "bayes",
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "model_name": {"value": "bert-base-uncased"},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "batch_size": {"values": [16, 32, 64]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
        "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "epochs": {"value": 5},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 2},
}

sweep_id = wandb.sweep(sweep_config, project="transformer-classification")
# Launch 30 agents to explore the parameter space
wandb.agent(sweep_id, function=train_one_run, count=30)
```

---

### Experiment Tracking Comparison

| Feature | MLflow | Weights & Biases | Neptune |
|---------|--------|-----------------|---------|
| **Self-hosted** | Yes (free, Apache 2) | Yes (enterprise only) | Yes (enterprise only) |
| **Cloud SaaS** | Databricks-managed | Yes (generous free tier) | Yes (free tier) |
| **UI quality** | Good | Excellent | Very good |
| **Hyperparameter sweeps** | Via Optuna plugin | Native Bayesian/random/grid | Native |
| **LLM/prompt tracking** | MLflow AI Gateway | W&B Prompts (beta) | Neptune |
| **Dataset versioning** | Artefacts only | W&B Artifacts | Neptune Files |
| **Model registry** | Built-in, stage-based | Built-in | Built-in |
| **Integrations** | 20+ | 50+ | 25+ |
| **Best for** | On-premise / compliance | Research teams, rich sweeps | Enterprise audit trails |

---

## Feature Stores

### Online vs Offline Stores

A feature store decouples feature computation from model training and serving, making features consistently available to both pipelines.

| Dimension | Offline Store | Online Store |
|-----------|--------------|-------------|
| **Purpose** | Training data generation | Real-time serving at inference |
| **Storage backend** | Data warehouse (BigQuery, Redshift, S3+Parquet) | Redis, DynamoDB, Bigtable |
| **Latency** | Minutes to hours (batch) | < 10 ms |
| **Update frequency** | Hourly or daily batch jobs | Streaming or near-real-time |
| **Data volume** | Full historical records (petabytes) | Latest value per entity |

### The Point-in-Time Correctness Problem

The most common source of training-serving skew is **temporal leakage** in training data construction. When building a training dataset, joining features naively on entity ID pulls the *current* feature value, not the value that existed *at the time of the label*.

```
Example: credit fraud model
  t=0   → transaction occurs (inference time: features must reflect t=0 state)
  t=7   → account flagged suspicious (account_flagged = True)
  t=30  → fraud confirmed → label generated for training

Naive join: training row gets account_flagged = True  ← LEAKAGE
Point-in-time join: training row gets account_flagged = False  ← CORRECT
```

### Feast Setup

```python
"""
feast_features.py — Point-in-time correct feature store with Feast.

Requirements:
    pip install feast[redis]>=0.38
"""
from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureService, FeatureStore, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float32, Int64, String

# ── Entity ───────────────────────────────────────────────────────────────────
user = Entity(name="user_id", description="Unique user identifier")

# ── Data source ──────────────────────────────────────────────────────────────
user_stats_source = FileSource(
    path="data/user_stats.parquet",          # can also be BigQuery, Redshift
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ── Feature view ─────────────────────────────────────────────────────────────
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="purchase_count_7d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float32),
        Field(name="days_since_last_login", dtype=Int64),
        Field(name="preferred_category", dtype=String),
    ],
    online=True,
    source=user_stats_source,
)

# ── Feature service (named set for a model) ──────────────────────────────────
recommendation_svc = FeatureService(
    name="recommendation_v1",
    features=[
        user_stats_fv[["purchase_count_7d", "avg_order_value_30d"]],
    ],
)
```

```python
from datetime import datetime
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Materialise to online store (incremental, safe to run repeatedly)
store.materialize_incremental(end_date=datetime.utcnow())

# ── Online serving (< 10 ms with Redis) ──────────────────────────────────────
online_features = store.get_online_features(
    features=[
        "user_stats:purchase_count_7d",
        "user_stats:avg_order_value_30d",
        "user_stats:days_since_last_login",
    ],
    entity_rows=[
        {"user_id": "user_42"},
        {"user_id": "user_99"},
    ],
).to_dict()

# ── Point-in-time correct historical features for training ───────────────────
entity_df = pd.DataFrame(
    {
        "user_id": ["user_42", "user_99", "user_07"],
        "event_timestamp": pd.to_datetime(
            ["2024-01-15", "2024-01-16", "2024-01-14"]
        ),
        "label": [1, 0, 1],
    }
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_stats:purchase_count_7d",
        "user_stats:avg_order_value_30d",
        "user_stats:days_since_last_login",
    ],
).to_df()
```

---

## Data Versioning with DVC

DVC (Data Version Control) brings Git-like versioning to large data files and defines ML pipelines as dependency-tracked DAGs. Large files live in remote storage (S3, GCS, Azure Blob); lightweight `.dvc` pointer files are committed to Git.

```bash
# ── Initialise ────────────────────────────────────────────────────────────────
pip install dvc dvc-s3
git init && dvc init

# Configure remote storage
dvc remote add -d myremote s3://my-ml-bucket/dvc-store
dvc remote modify myremote region us-east-1
git commit -m "Initialise DVC with S3 remote"

# ── Track a dataset ──────────────────────────────────────────────────────────
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc data/raw/.gitignore
git commit -m "Track raw training data v1"
dvc push                  # upload to S3

# ── On a new machine ─────────────────────────────────────────────────────────
git clone https://github.com/org/my-ml-project
dvc pull                  # download the data pinned by the commit
```

**Defining a pipeline as a DAG in `dvc.yaml`:**

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py --input data/raw/train.csv --output data/processed/
    deps:
      - src/preprocess.py
      - data/raw/train.csv
    outs:
      - data/processed/train_features.parquet
      - data/processed/val_features.parquet

  train:
    cmd: python src/train.py --data data/processed/ --model models/model.pkl
    deps:
      - src/train.py
      - data/processed/train_features.parquet
      - data/processed/val_features.parquet
    params:
      - params.yaml:
          - model.n_estimators
          - model.max_depth
          - training.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/eval.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py --model models/model.pkl --data data/processed/val_features.parquet
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/processed/val_features.parquet
    metrics:
      - metrics/eval.json:
          cache: false
```

```bash
dvc repro                 # only re-runs stages whose inputs changed
dvc metrics show          # display current metrics
dvc metrics diff HEAD~1   # compare metrics against previous commit
```

---

## Container Fundamentals for ML

### Multi-Stage Dockerfile for a Python ML App

Multi-stage builds separate the build environment (with compilers and dev dependencies) from the lean runtime environment. This keeps production images small and avoids shipping build tools.

```dockerfile
# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install into /install prefix so they can be copied in isolation
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy only the installed packages — no build tools
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY models/ ./models/

# Security: run as non-root
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

EXPOSE 8080
ENTRYPOINT ["python", "-m", "src.serve"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### GPU-Enabled Base Image (CUDA)

For LLM inference and deep learning training, base on NVIDIA's official CUDA images. The `cudnn-runtime` variant includes cuDNN without training-time compiler tools.

```dockerfile
# GPU runtime image: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY src/ ./src/
COPY models/ ./models/

# Expose only GPU 0 by default; override at runtime
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8080
ENTRYPOINT ["python", "src/serve_gpu.py"]
```

```bash
# Build
DOCKER_BUILDKIT=1 docker build --target runtime -t my-ml-app:latest .

# Run with GPU pass-through (requires nvidia-container-toolkit)
docker run --gpus all -p 8080:8080 my-ml-app:latest

# Verify GPU access inside the container
docker run --gpus all my-ml-app:latest \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Kubernetes for ML

### Pod Spec with GPU Resource Limits

```yaml
# llm-inference-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
  labels:
    app: llm-inference
spec:
  containers:
    - name: inference-server
      image: my-registry/llm-serve:1.2.0
      ports:
        - containerPort: 8080
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "8"
          memory: "32Gi"
          nvidia.com/gpu: "1"
      env:
        - name: MODEL_ID
          value: "meta-llama/Llama-3-8B-Instruct"
        - name: MAX_BATCH_SIZE
          value: "32"
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 60
        periodSeconds: 15
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
  nodeSelector:
    accelerator: nvidia-a100
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### Deployment and Service

```yaml
# llm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0          # zero-downtime rolling update
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
        - name: inference-server
          image: my-registry/llm-serve:1.2.0
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-svc
  namespace: ml-serving
spec:
  selector:
    app: llm-inference
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

### Horizontal Pod Autoscaler

```yaml
# llm-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

---

## Cloud ML Platform Comparison

| Capability | AWS SageMaker | Google Vertex AI | Azure ML |
|------------|--------------|-----------------|---------|
| **Managed training jobs** | SageMaker Training | Custom Training Jobs | Compute Clusters |
| **Spot / preemptible** | Managed Spot Training | Preemptible VMs | Low-Priority Compute |
| **Distributed training** | SageMaker Distributed (data + model parallel) | Vertex AI Distributed | DeepSpeed / Horovod |
| **Model registry** | SageMaker Model Registry | Vertex AI Model Registry | Azure ML Model Registry |
| **Real-time endpoints** | SageMaker Real-time Inference | Vertex AI Online Prediction | Azure ML Online Endpoints |
| **Batch transform** | SageMaker Batch Transform | Vertex AI Batch Prediction | Azure ML Batch Endpoints |
| **Feature store** | SageMaker Feature Store | Vertex AI Feature Store | Azure ML (limited) |
| **Experiment tracking** | SageMaker Experiments | Vertex AI Experiments | Native MLflow integration |
| **AutoML** | SageMaker Autopilot | Vertex AI AutoML | Azure AutoML |
| **Pipeline orchestration** | SageMaker Pipelines | Vertex AI Pipelines (Kubeflow) | Azure ML Pipelines |
| **LLM fine-tuning** | SageMaker JumpStart | Vertex AI Model Garden | Azure AI Studio |
| **Pricing model** | Per-second compute + storage | Per-second compute + storage | Per-minute compute + storage |

!!! tip "When to use a managed platform"
    Choose a managed cloud platform when your team lacks Kubernetes expertise, when deep integration with cloud data services is required, or when compliance controls (VPC, IAM, encryption at rest) are mandated. Choose self-hosted (Kubernetes + MLflow + Feast) when multi-cloud flexibility, avoiding vendor lock-in, or raw cost optimisation is the priority.

---

## Exercises

1. **MLflow sweep**: Run a nine-run hyperparameter grid over `n_estimators` ∈ {100, 200, 400} and `max_depth` ∈ {3, 5, 7} using the `train_and_track` function from this chapter. Use `mlflow.search_runs()` to programmatically find the run with the highest `roc_auc` and transition it to the `Production` stage in the model registry.

2. **W&B Bayesian sweep**: Define a W&B sweep with at least five hyperparameters (include at least one log-uniform continuous parameter). Run 20 agents and use the W&B parallel coordinates plot to identify the top-3 most influential parameters. Write a 200-word interpretation of your findings.

3. **Feature store simulation**: Without installing Feast, implement a `PointInTimeFeatureStore` class in Python. It should store features as a `pd.DataFrame` keyed by `(entity_id, timestamp)` and implement `get_historical_features(entity_df: pd.DataFrame) -> pd.DataFrame` with correct point-in-time semantics using `pd.merge_asof`.

4. **DVC pipeline**: Convert an existing three-step ML workflow (preprocess → train → evaluate) into a DVC pipeline. Add a `params.yaml` for two hyperparameters. Verify that (a) `dvc repro` is a no-op when nothing changes, (b) changing a parameter re-runs only downstream stages, and (c) `dvc metrics diff` shows the metric change.

5. **Kubernetes HPA design**: Write a Kubernetes Deployment, Service, and HPA spec for a BERT-based classification model. The HPA should target p95 latency < 200 ms (assume a custom metrics adapter that exposes `http_p95_latency_ms`). Set appropriate `minReplicas`, `maxReplicas`, `scaleUp`, and `scaleDown` behaviour, and justify each value.

---

## Summary

This chapter established the infrastructure layer that underpins reliable ML systems:

- **Experiment tracking** with MLflow (open-source, self-hosted) and W&B (cloud-native, richer sweep tooling) prevents the "which run was that?" problem and supports systematic hyperparameter exploration.
- **Feature stores** with dual offline/online architectures solve training-serving skew and guarantee point-in-time correctness in training data — the single most important correctness guarantee in production ML.
- **DVC** treats data files and pipelines as versioned artefacts, enabling any experiment to be reproduced from a Git commit hash alone.
- **Multi-stage Docker** images with CUDA-enabled GPU base images package ML workloads into portable, production-safe containers.
- **Kubernetes** provides the orchestration layer for scaling, rolling updates, GPU scheduling, and resource isolation — essential for multi-tenant ML serving.
- **Cloud ML platforms** (SageMaker, Vertex AI, Azure ML) offer managed versions of this infrastructure at the cost of reduced flexibility and potential vendor lock-in.

*Next: [Ch 2 — Model Deployment](../ch02-deployment/index.md)*
