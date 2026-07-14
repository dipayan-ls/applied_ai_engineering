---
title: "Ch 3 — ML Monitoring & Observability"
---

# Ch 3 — ML Monitoring & Observability

!!! info "Chapter Meta"
    **Level:** Advanced &nbsp;|&nbsp; **Reading time:** 75 min &nbsp;|&nbsp; **Volume:** 9 — MLOps

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain why ML monitoring requires fundamentally different practices from traditional software monitoring, specifically regarding silent degradation and delayed ground truth.
2. Implement PSI, Kolmogorov-Smirnov, and chi-squared tests to detect data drift in production feature distributions.
3. Monitor LLM-specific quality metrics (hallucination rate, faithfulness, toxicity, cost per query) using Langfuse tracing and LLM-as-judge scoring.
4. Configure Prometheus alert rules and Alertmanager routing for infrastructure and drift metrics.
5. Design a composite retraining trigger policy combining scheduled, drift-triggered, and performance-triggered strategies.

---

## Why ML Monitoring Differs from Software Monitoring

Traditional software fails loudly: an exception is thrown, an error code is returned, and a monitoring dashboard turns red. ML systems fail silently:

- **No clear error signal**: A model that classifies 60% of fraud correctly (down from 85%) still returns valid HTTP 200 responses. Nothing breaks — the model is simply wrong more often.
- **Silent degradation**: Distribution shifts accumulate gradually. By the time business impact is measurable, weeks of damage have occurred.
- **Delayed labels**: The ground truth for a loan default may arrive 60 days after the prediction. Accuracy cannot be computed in real time; only proxy metrics are available immediately.

```
Timeline for a loan default model:
  Day 0   → model predicts "no default" (confidence 0.82)
  Day 5   → first missed payment (observable signal)
  Day 60  → loan formally defaulted (ground-truth label)
  Day 90  → model retrained — 3 months of degraded predictions affected decisions
```

Effective ML monitoring bridges the gap between deployment and delayed ground truth using proxy metrics, statistical drift detection, and LLM-as-judge quality scoring.

---

## Infrastructure Metrics

Before monitoring model quality, instrument the serving layer. Infrastructure failures are detectable immediately and affect all users simultaneously.

| Metric | Description | Target threshold |
|--------|-------------|-----------------|
| **Latency p50** | Median end-to-end response time | < 300 ms for LLM TTFT |
| **Latency p95** | 95th percentile response time | < 2 s SLO (typical) |
| **Latency p99** | 99th percentile response time | < 5 s (long-tail budget) |
| **Throughput (req/s)** | Requests per second served | Capacity-plan dependent |
| **GPU utilisation** | % GPU compute in use | 60–80% (headroom for spikes) |
| **GPU memory utilisation** | % VRAM occupied | < 90% (headroom for KV cache) |
| **Error rate** | % of requests with 4xx/5xx status | < 0.1% |
| **Cost per query** | USD per API request | Budget-dependent |

Expose these from FastAPI using `prometheus-fastapi-instrumentator`:

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

app = FastAPI()

# Custom LLM-specific counters
token_counter = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    labelnames=["model", "direction"],    # direction: input | output
)
cost_histogram = Histogram(
    "llm_request_cost_usd",
    "Cost per LLM request in USD",
    buckets=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
)
gpu_utilisation_gauge = Gauge("gpu_utilisation_percent", "GPU utilisation %")

# Instrument all HTTP endpoints automatically
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

---

## Data Drift Detection

Data drift occurs when the distribution of model inputs in production diverges from the training distribution. It does not require labels to detect.

### PSI — Population Stability Index

PSI is the most widely used metric for monitoring numeric feature distributions in production ML, especially in financial services.

$$\text{PSI} = \sum_{i=1}^{n} (A_i - E_i) \ln\!\left(\frac{A_i}{E_i}\right)$$

Where $E_i$ is the proportion in bin $i$ in the reference (training) distribution and $A_i$ is the proportion in the current (production) distribution.

**Interpretation thresholds:**

| PSI | Interpretation | Recommended action |
|-----|---------------|-------------------|
| < 0.10 | No significant shift | Monitor normally |
| 0.10 – 0.20 | Moderate shift | Investigate feature pipeline; increase monitoring frequency |
| > 0.20 | Significant shift | Investigate immediately; consider model refresh |

```python
"""
drift_detection.py — Statistical drift tests for ML monitoring.

Requirements:
    pip install numpy scipy pandas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between two 1-D numeric distributions.

    Args:
        expected: Reference distribution (e.g., training feature values).
        actual:   Current distribution (e.g., production feature values).
        n_bins:   Number of quantile-based bins.
        epsilon:  Smoothing term to avoid log(0).

    Returns:
        PSI value (float). Threshold: < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant.
    """
    bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    e_pct = expected_counts / len(expected) + epsilon
    a_pct = actual_counts / len(actual) + epsilon

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return psi
```

### KS Test for Continuous Features

The two-sample Kolmogorov-Smirnov test measures the maximum absolute difference between two empirical CDFs. It is non-parametric and distribution-free.

```python
def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sample KS test for distribution shift on a continuous feature.

    Returns:
        dict with statistic, p_value, and drift_detected flag.
    """
    result = stats.ks_2samp(reference, current)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drift_detected": bool(result.pvalue < alpha),
        "interpretation": (
            "Distributions differ significantly" if result.pvalue < alpha
            else "No significant difference detected"
        ),
    }
```

### Chi-Squared Test for Categorical Features

For categorical features, the chi-squared test detects whether the observed category frequencies differ significantly from the expected (reference) frequencies.

```python
from scipy.stats import chi2_contingency


def chi2_drift_test(
    reference_series: pd.Series,
    current_series: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared test for distribution shift on a categorical feature.

    Args:
        reference_series: Categorical values from the reference period.
        current_series:   Categorical values from the current period.
        alpha:            Significance level (default 5%).

    Returns:
        dict with chi2 statistic, p_value, degrees of freedom, drift_detected.
    """
    categories = set(reference_series.unique()) | set(current_series.unique())
    ref_counts = reference_series.value_counts().reindex(categories, fill_value=0)
    cur_counts = current_series.value_counts().reindex(categories, fill_value=0)

    contingency = pd.DataFrame({"reference": ref_counts, "current": cur_counts})
    chi2, p_value, dof, _ = chi2_contingency(contingency.T)

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "drift_detected": bool(p_value < alpha),
    }
```

### Drift Monitoring with Evidently AI

Evidently AI provides pre-built drift tests and HTML reports for the full feature set:

```python
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

reference_data = pd.read_parquet("data/reference_features.parquet")
production_data = pd.read_parquet("data/production_week_24.parquet")

report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference_data, current_data=production_data)
report.save_html("drift_report_week_24.html")

result = report.as_dict()
drifted = [
    m["result"]["column_name"]
    for m in result["metrics"]
    if m.get("result", {}).get("drift_detected", False)
]
print(f"Drifted features ({len(drifted)}): {drifted}")
if len(drifted) >= 3:
    trigger_retraining_pipeline()
```

---

## Concept Drift

Concept drift occurs when the statistical relationship between inputs and the correct output changes — the model's mapping becomes stale even if input distributions remain stable.

### Outcome Drift (Target Distribution Changes)

If the base rate of positive labels changes (e.g., fraud becomes more common due to a new attack vector), a model calibrated for the old rate will be systematically miscalibrated.

Monitor the distribution of model output scores (not just accuracy) as a proxy:

```python
def detect_prediction_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Detect drift in the distribution of model prediction probabilities.
    A shift here indicates concept drift even without ground-truth labels.
    """
    return ks_drift_test(reference_scores, current_scores, alpha=alpha)
```

### Delayed Label Collection Problem

| Strategy | Latency | Cost | Accuracy |
|----------|---------|------|----------|
| **Proxy labels** (e.g., click = implicit positive) | Immediate | Low | Noisy |
| **Human spot-checking** (1–5% sample) | Same day | High | High |
| **LLM-as-judge** | Minutes | Medium | Medium-High |
| **A/B testing with outcome metrics** | Weeks | Low | High |
| **Weak supervision (Snorkel)** | Hours | Low | Medium |

### Surrogate Metrics

When ground-truth labels are unavailable, use leading indicators that correlate with future model quality:

- **Prediction confidence distribution**: a sharp drop in mean confidence often precedes accuracy degradation.
- **Feature anomaly rate**: % of requests where any feature is outside the training range.
- **Output length distribution** (for LLMs): sudden shifts in token count can indicate prompt or context drift.

---

## LLM-Specific Monitoring

LLMs exhibit failure modes — hallucination, factual drift, instruction non-compliance — that accuracy metrics do not capture.

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Hallucination rate** | % outputs containing unsupported factual claims | LLM-as-judge + human audit |
| **Faithfulness** | Fraction of answer grounded in retrieved context | LLM-as-judge (0–1 scale) |
| **Relevance score** | How well the answer addresses the question | LLM-as-judge |
| **Toxicity** | % outputs containing harmful/offensive content | Classifier (e.g., `detoxify`) |
| **Cost per query** | USD spent per request | `usage.input_tokens × price + usage.output_tokens × price` |
| **Latency per token** | ms / output token | `latency_ms / tokens_generated` |

---

## Langfuse for LLM Observability

Langfuse is an open-source observability platform for LLM applications providing distributed tracing, quality scoring, cost tracking, and dashboards.

```python
"""
llm_observability.py — LLM tracing and quality monitoring with Langfuse.

Requirements:
    pip install langfuse anthropic
"""
from __future__ import annotations

import json
import os

import anthropic
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)
client = anthropic.Anthropic()

FAITHFULNESS_PROMPT = """\
You evaluate whether an AI answer is faithfully grounded in the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Rate faithfulness 0.0–1.0 (1.0 = fully grounded, 0.0 = fabricated).
Respond with JSON only: {{"faithfulness": <float>, "reason": "<brief>"}}
"""


@observe(name="rag_query")
def rag_with_observability(question: str, context: str) -> str:
    """
    Execute a RAG query with full Langfuse tracing and quality scoring.

    Traces: input/output, token usage, cost, model
    Scores: faithfulness (LLM-as-judge), latency_per_token
    """
    langfuse_context.update_current_trace(
        name="RAG Pipeline",
        input={"question": question, "context_length": len(context)},
        tags=["rag", "production"],
    )

    # ── Generation ────────────────────────────────────────────────────────────
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the provided context. "
                    "If the context does not contain the answer, say so."
                ),
            }
        ],
    )
    answer = response.content[0].text

    # ── Faithfulness scoring (LLM-as-judge) ──────────────────────────────────
    judge_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": FAITHFULNESS_PROMPT.format(
                    context=context, question=question, answer=answer
                ),
            }
        ],
    )

    faithfulness_score = 0.5
    try:
        score_data = json.loads(judge_response.content[0].text)
        faithfulness_score = float(score_data.get("faithfulness", 0.5))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # ── Cost tracking ─────────────────────────────────────────────────────────
    input_cost = response.usage.input_tokens * 0.80 / 1_000_000     # $/token (Haiku)
    output_cost = response.usage.output_tokens * 4.00 / 1_000_000
    total_cost_usd = input_cost + output_cost

    # ── Log scores to Langfuse ────────────────────────────────────────────────
    langfuse_context.score_current_trace(
        name="faithfulness",
        value=faithfulness_score,
        comment=f"cost_usd={total_cost_usd:.6f}",
    )
    langfuse_context.update_current_trace(
        output={"answer": answer},
        metadata={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": total_cost_usd,
        },
    )

    return answer
```

---

## Alerting

### Prometheus + Grafana Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/ml_alerts.yml"

scrape_configs:
  - job_name: llm-serving
    static_configs:
      - targets: ["llm-server:8080"]
    metrics_path: /metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]
```

### Alert Rules in YAML

```yaml
# alerts/ml_alerts.yml
groups:
  - name: ml-serving
    rules:
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m]) /
          rate(http_requests_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "Error rate > 1% for 2 minutes on {{ $labels.job }}"
          runbook: "https://wiki.example.com/runbooks/ml-high-error-rate"

      - alert: HighP95Latency
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 latency > 2 s for 5 minutes"

      - alert: DataDriftDetected
        expr: ml_psi_score{} > 0.2
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "PSI > 0.2 on feature {{ $labels.feature }} — possible data drift"

      - alert: LowFaithfulnessScore
        expr: avg_over_time(llm_faithfulness_score[30m]) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Mean faithfulness < 0.7 — possible hallucination regression"

      - alert: HighGPUMemoryPressure
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.92
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory > 92% — OOM risk"
```

### PagerDuty Integration Pattern

```yaml
# alertmanager.yml
route:
  group_by: ["alertname", "severity"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: "slack-default"
  routes:
    - match:
        severity: critical
      receiver: "pagerduty-ml-oncall"
      continue: true

receivers:
  - name: "slack-default"
    slack_configs:
      - api_url: "$SLACK_WEBHOOK_URL"
        channel: "#ml-alerts"
        title: "{{ .GroupLabels.alertname }}"
        text: "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"

  - name: "pagerduty-ml-oncall"
    pagerduty_configs:
      - routing_key: "$PAGERDUTY_ROUTING_KEY"
        description: "{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}"
        severity: "{{ .GroupLabels.severity }}"
        details:
          runbook: "{{ .CommonAnnotations.runbook }}"
```

---

## Retraining Triggers

No single trigger strategy is sufficient. Combine all three:

### Scheduled Retraining

Retrain on a fixed cadence regardless of observed drift. Simple; provides a guaranteed freshness floor.

```python
# Example: weekly retraining every Sunday at 03:00 UTC
# In Kubernetes, use a CronJob:
# schedule: "0 3 * * 0"
```

### Drift-Triggered (PSI Threshold)

```python
def should_retrain_on_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    psi_threshold: float = 0.2,
    n_features_to_trigger: int = 3,
) -> tuple[bool, list[str]]:
    """
    Trigger retraining when PSI exceeds threshold for at least
    `n_features_to_trigger` numeric features.

    Returns:
        (trigger_retraining, list_of_drifted_features)
    """
    drifted: list[str] = []
    for col in reference_data.select_dtypes(include="number").columns:
        psi = compute_psi(reference_data[col].values, current_data[col].values)
        if psi > psi_threshold:
            drifted.append(col)
    return len(drifted) >= n_features_to_trigger, drifted
```

### Performance-Triggered (Metric Threshold)

```python
def should_retrain_on_performance(
    recent_metric: float,
    baseline_metric: float,
    relative_drop_threshold: float = 0.05,
    metric_name: str = "accuracy",
) -> bool:
    """
    Trigger retraining when a quality metric drops by more than
    `relative_drop_threshold` relative to baseline.

    Args:
        recent_metric:           Current period metric value.
        baseline_metric:         Reference (post-last-training) metric value.
        relative_drop_threshold: Fraction drop to trigger retraining (default 5%).

    Returns:
        True if retraining should be triggered.
    """
    relative_drop = (baseline_metric - recent_metric) / (baseline_metric + 1e-9)
    if relative_drop > relative_drop_threshold:
        print(
            f"RETRAIN: {metric_name} dropped {relative_drop:.1%} "
            f"({baseline_metric:.4f} → {recent_metric:.4f})"
        )
        return True
    return False
```

!!! tip "Combined trigger policy"
    Run drift-triggered checks daily. Run performance-triggered checks as soon as labels arrive (even partial). Keep scheduled retraining as a weekly backstop. Any one trigger is sufficient to initiate the retraining pipeline.

---

## Exercises

1. **PSI monitoring pipeline**: Write a script that (a) loads a reference feature distribution from a Parquet file, (b) loads 12 weekly production snapshots, (c) computes PSI for each numeric feature per week, (d) plots a PSI time series heatmap (weeks × features), and (e) prints an alert for any week where PSI > 0.2 for three or more features.

2. **LLM faithfulness + relevance monitor**: Extend `rag_with_observability` to also score **relevance** (does the answer address the question?) using a second LLM-as-judge call. Log both scores to Langfuse and plot the distribution of faithfulness and relevance scores over 100 test queries.

3. **Prometheus alert**: Write a Prometheus alert rule that fires when the 30-minute rolling mean of `llm_faithfulness_score` drops below 0.70. Write the Alertmanager route to send a Slack message to `#ml-quality` and a PagerDuty alert for severity `critical`. Test the alert by pushing a low score via `prometheus_client`.

4. **Retraining policy simulation**: Simulate 26 weekly snapshots with linearly increasing PSI (0.05 → 0.35). Implement the three trigger strategies from this chapter. Compare: number of retraining events, weeks of degraded model operation (PSI > 0.2), and compute cost (assume 10 GPU-hours per retraining) across all three strategies.

5. **Evidently automated report**: Use Evidently AI to build a weekly monitoring job that (a) loads the current week's production features, (b) runs drift detection against the reference, (c) saves the HTML report to S3, and (d) sends a summary email via `smtplib` listing the drifted features and their PSI values.

---

## Summary

- **ML monitoring differs fundamentally** from software monitoring: silent degradation, no clear error signal, and delayed labels mean that standard alerting on exceptions misses most ML quality regressions.
- **Infrastructure metrics** (p50/p95/p99 latency, throughput, GPU utilisation, error rate) should be instrumented first — they fail loudly and are immediately actionable.
- **Data drift detection** uses PSI ($< 0.1$ stable, $0.1$–$0.2$ moderate, $> 0.2$ significant) for numeric features, KS test for distributional comparison, and chi-squared for categorical features — all implementable without ground-truth labels.
- **Concept drift** manifests as changes in outcome distributions or prediction confidence; it is detected through surrogate metrics, LLM-as-judge scoring, or delayed ground-truth comparison.
- **LLM-specific monitoring** — faithfulness, hallucination rate, toxicity, cost per query, latency per token — requires an LLM-as-judge pipeline; Langfuse provides the tracing infrastructure to make this operationally sustainable.
- **Retraining triggers** should combine scheduled (freshness guarantee), drift-triggered (proactive), and performance-triggered (outcome-based) strategies — each catches a different class of model degradation.

*Next: [Volume 10 — Enterprise AI](../../volume10-enterprise-ai/index.md)*
