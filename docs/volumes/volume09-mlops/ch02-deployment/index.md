---
title: "Ch 2 — Model Deployment"
---

# Ch 2 — Model Deployment

!!! info "Chapter Meta"
    **Level:** Advanced &nbsp;|&nbsp; **Reading time:** 90 min &nbsp;|&nbsp; **Volume:** 9 — MLOps

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Select the appropriate serving pattern — online, batch, or streaming — for a given ML use case based on latency requirements, throughput, and cost constraints.
2. Build a production-ready LLM serving endpoint with FastAPI, including async streaming via Server-Sent Events, Pydantic validation, and JWT authentication middleware.
3. Export a model to ONNX or TorchScript and understand when to use SafeTensors for safe, cross-framework weight storage.
4. Apply quantisation (INT8, INT4/GPTQ), continuous batching, Flash Attention, and KV cache management to reduce LLM inference cost and latency.
5. Implement blue-green and canary deployment strategies with traffic splitting, monitoring hooks, and automated rollback procedures.

---

## Serving Patterns

Choosing the right serving pattern before building infrastructure saves weeks of rework. The three fundamental patterns each make different trade-offs across latency, throughput, and cost:

| Dimension | Online (Real-time) | Batch | Streaming |
|-----------|-------------------|-------|-----------|
| **Latency** | < 100–500 ms (p99) | Minutes to hours | 1–5 s (near-real-time) |
| **Throughput** | Low to medium req/s | Very high (millions/run) | Medium events/s |
| **Trigger** | Synchronous HTTP request | Scheduled job / pipeline event | Message queue (Kafka, Kinesis) |
| **Infrastructure** | Load-balanced REST API | Batch compute (Spark, Ray) | Stream processor (Flink, Spark Streaming) |
| **Use cases** | Fraud detection, chatbots, search ranking | Nightly churn scoring, bulk embeddings | Real-time recommendations, anomaly detection on event streams |
| **Cost model** | Pay for peak capacity (always-on GPUs) | Pay only for compute time | Pay for stream processing + compute |

!!! warning "Latency Cliff"
    Online serving at p99 < 100 ms leaves almost no room for model loading, serialisation overhead, or remote feature lookup. Profile the *full* request path — not just model inference time — before committing to a latency SLO.

---

## REST API with FastAPI for LLM Serving

The following production endpoint serves an LLM with streaming output, JWT authentication, Pydantic validation, and async handling. Every line is typed and PEP 8 compliant.

```python
"""
serve.py — Production LLM serving endpoint with FastAPI.

Requirements:
    pip install fastapi uvicorn[standard] pydantic python-jose[cryptography] vllm
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)
security = HTTPBearer()

# ── Auth ──────────────────────────────────────────────────────────────────────
SECRET_KEY: str = os.environ["JWT_SECRET_KEY"]
ALGORITHM: str = "HS256"


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate JWT bearer token; raise 401 on failure."""
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ── Request / response models ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float


# ── App + engine ──────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

engine: AsyncLLMEngine | None = None


@app.on_event("startup")
async def load_model() -> None:
    global engine
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=os.environ.get("MODEL_ID", "meta-llama/Llama-3-8B-Instruct"),
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )
    )
    logger.info("vLLM engine ready")


# ── SSE streaming helper ──────────────────────────────────────────────────────
async def token_stream(
    request_id: str,
    prompt: str,
    params: SamplingParams,
) -> AsyncIterator[str]:
    async for output in engine.generate(prompt, params, request_id):  # type: ignore[union-attr]
        for completion in output.outputs:
            yield f"data: {json.dumps({'token': completion.text})}\n\n"
    yield "data: [DONE]\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    claims: dict = Depends(verify_token),
) -> GenerateResponse | StreamingResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    params = SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
    )
    request_id = f"{claims.get('sub', 'anon')}-{time.monotonic_ns()}"

    if body.stream:
        return StreamingResponse(
            token_stream(request_id, body.prompt, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    t0 = time.monotonic()
    outputs = []
    async for out in engine.generate(body.prompt, params, request_id):
        outputs = out.outputs
    text = outputs[0].text if outputs else ""
    return GenerateResponse(
        text=text,
        tokens_generated=len(outputs[0].token_ids) if outputs else 0,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": engine is not None}
```

```bash
# Single Uvicorn worker per GPU — vLLM handles concurrency internally
uvicorn serve:app --host 0.0.0.0 --port 8080 --loop uvloop
```

---

## Model Serialisation

### ONNX Export from PyTorch

ONNX produces a hardware-agnostic graph representation that ONNX Runtime can execute with CPU or GPU-specific optimisations (TensorRT, OpenVINO).

```python
"""
export_onnx.py — Export a PyTorch classifier to ONNX with dynamic batch size.

Requirements:
    pip install torch onnx onnxruntime
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "model.onnx",
    opset_version: int = 17,
) -> None:
    """Export model to ONNX with dynamic batch size and sequence length."""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
    print(f"Exported ONNX model → {output_path}")


def run_onnx_inference(
    model_path: str,
    input_ids: np.ndarray,
) -> np.ndarray:
    """Run inference with ONNX Runtime (CUDA preferred, CPU fallback)."""
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return session.run(output_names=["logits"], input_feed={"input_ids": input_ids})[0]
```

### TorchScript (`torch.jit.trace`)

TorchScript compiles a model to a static graph, enabling deployment without a Python runtime and C++ embedding.

```python
import torch
import torch.nn as nn


def export_torchscript(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "model_scripted.pt",
) -> None:
    """Trace model to TorchScript for C++ or mobile deployment."""
    model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(model, sample_input)
    scripted.save(output_path)


def load_torchscript(model_path: str, x: torch.Tensor) -> torch.Tensor:
    loaded = torch.jit.load(model_path)
    loaded.eval()
    with torch.no_grad():
        return loaded(x)
```

### SafeTensors

SafeTensors is Hugging Face's format for model weights. It is safe (no `pickle`, no code execution on load), fast (memory-mapped), and cross-framework. It is now the default for models on the Hugging Face Hub.

```python
from safetensors.torch import load_file, save_file
import torch

# Save
save_file(model.state_dict(), "model.safetensors")

# Load — zero-copy memory map, fast and safe
state_dict = load_file("model.safetensors", device="cuda")
model.load_state_dict(state_dict)
```

---

## Inference Optimisation Techniques

### Quantisation

Quantisation reduces the numerical precision of weights, shrinking GPU memory footprint and increasing throughput.

**INT8 weight quantisation (PyTorch):**

```python
import torch
from torch.quantization import quantize_dynamic

# Dynamically quantise Linear layers to INT8
quantised = quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8,
)

# Memory comparison
fp32_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
int8_mb = sum(p.numel() * 1 for p in model.parameters()) / 1e6
print(f"FP32: {fp32_mb:.0f} MB → INT8: {int8_mb:.0f} MB  (4× reduction)")
```

**INT4 quantisation with GPTQ:**

GPTQ (Generalised Post-Training Quantisation) compresses LLM weights to 4 bits using layer-wise second-order weight correction on a calibration dataset.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
tokeniser = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    quantize_config=quant_config,
)

# Calibration data (representative production queries)
calibration = [
    tokeniser("Summarise the following document.", return_tensors="pt").input_ids,
    tokeniser("What are the key risks?", return_tensors="pt").input_ids,
]
model.quantize(calibration)
model.save_quantized("llama3-8b-int4-gptq")
```

**Memory savings for Llama-3-8B:**

| Precision | Bytes/param | Model memory | Reduction vs BF16 |
|-----------|------------|-------------|-------------------|
| FP32 | 4 | 32 GB | — |
| BF16 | 2 | 16 GB | baseline |
| INT8 | 1 | 8 GB | 2× |
| INT4 (GPTQ) | 0.5 | 4 GB | 4× |

---

### Continuous Batching

Naive static batching pads all sequences in a batch to the longest, wasting compute on padding tokens. **Continuous batching** (iteration-level scheduling) evicts completed sequences and inserts new requests into the freed KV cache slots at each decode step — no padding required.

```
Static batching (batch of 4):
  Lengths: [128, 512, 64, 1024] → all padded to 1024
  Wasted FLOPs: 896 + 512 + 960 = 2,368 padding token steps

Continuous batching:
  At step 64: sequence 3 finishes → new request inserted
  At step 128: sequence 1 finishes → new request inserted
  GPU utilisation: ~90% vs ~40–60% for static batching
```

---

### Flash Attention

Standard attention materialises the full N×N attention matrix in HBM (high-bandwidth memory), requiring O(N²) memory. Flash Attention (Dao et al., 2022) tiles the computation in SRAM, reducing memory complexity to O(N) and eliminating redundant HBM reads/writes.

| Metric | Standard Attention | Flash Attention 2 |
|--------|-------------------|--------------------|
| Memory complexity | O(N²) | O(N) |
| HBM reads/writes | O(N²) | O(N) |
| Speed-up on A100 | 1× baseline | 2–4× |

```python
from transformers import AutoModelForCausalLM
import torch

# Flash Attention 2 — enabled via attn_implementation flag
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

---

### KV Cache

During autoregressive decoding, the Key (K) and Value (V) tensors from previous tokens are cached to avoid recomputation at each step. KV cache is the primary factor limiting maximum context length and batch size.

**KV cache memory formula:**

$$\text{KV cache (bytes)} = n_{\text{layers}} \times n_{\text{heads}} \times \text{seq\_len} \times d_{\text{head}} \times 2 \times \text{batch\_size} \times \text{dtype\_bytes}$$

**Example — Llama-3-8B (32 layers, 32 heads, head dim 128) at BF16, context 4096, batch 8:**

$$32 \times 32 \times 4096 \times 128 \times 2 \times 8 \times 2 = 8{,}589{,}934{,}592 \approx 8.6 \text{ GB}$$

On a 16 GB GPU with the BF16 model weight at 16 GB, this is not feasible — INT4 quantisation of weights (4 GB) makes it viable.

---

## vLLM: PagedAttention

vLLM implements **PagedAttention**, managing the KV cache in fixed-size pages (analogous to OS virtual memory paging). Benefits:

- Sequences of different lengths share GPU memory without pre-allocating worst-case contiguous blocks.
- KV cache utilisation approaches 100% of available HBM.
- Throughput is **3–24× higher** than naive HuggingFace `generate()` under concurrent load.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)

prompts = [
    "Explain the attention mechanism in transformers.",
    "What is MLOps?",
    "Summarise the EU AI Act in three bullet points.",
]
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(output.outputs[0].text[:100])
```

---

## Horizontal Scaling

### nginx Load Balancer Configuration

```nginx
# /etc/nginx/conf.d/llm-api.conf
upstream llm_backends {
    least_conn;                            # route to backend with fewest active connections
    server llm-pod-0.ml-serving:8080;
    server llm-pod-1.ml-serving:8080;
    server llm-pod-2.ml-serving:8080;
    keepalive 64;
}

server {
    listen 80;
    server_name api.example.internal;

    location /generate {
        proxy_pass         http://llm_backends;
        proxy_http_version 1.1;
        proxy_set_header   Connection "";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        proxy_buffering    off;            # required for SSE streaming
    }

    location /health {
        proxy_pass http://llm_backends;
        access_log off;
    }
}
```

### Kubernetes Deployment with HPA (Target Latency SLO)

Scale to maintain p95 latency < 2 s. The HPA reacts before the SLO is breached by scaling at 1.8 s:

```yaml
# llm-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 2
  maxReplicas: 16
  metrics:
    - type: Pods
      pods:
        metric:
          name: http_p95_latency_ms       # custom metric via Prometheus adapter
        target:
          type: AverageValue
          averageValue: "1800"            # 1.8 s — headroom before 2 s SLO
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
```

---

## Blue-Green Deployment

Blue-green deployment maintains two complete, identical environments. Traffic switches atomically. Rollback is a single command: flip the Service selector back to the stable version.

```yaml
# Service initially routes all traffic to blue (v1)
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-svc
  namespace: ml-serving
spec:
  selector:
    app: llm-inference
    version: blue
  ports:
    - port: 80
      targetPort: 8080
```

**Traffic switch (zero-downtime):**

```bash
# Validate green is healthy before switching
kubectl rollout status deployment/llm-inference-green -n ml-serving

# Atomic switch: all traffic from blue → green
kubectl patch svc llm-inference-svc -n ml-serving \
  --patch '{"spec": {"selector": {"version": "green"}}}'

# Verify new endpoints
kubectl get endpoints llm-inference-svc -n ml-serving
```

**Rollback procedure:**

```bash
# Revert to blue in < 5 seconds
kubectl patch svc llm-inference-svc -n ml-serving \
  --patch '{"spec": {"selector": {"version": "blue"}}}'

# Keep green running for 48 h in standby; then scale down
kubectl scale deployment/llm-inference-green --replicas=0 -n ml-serving
```

---

## Canary Deployment

Canary releases gradually shift traffic to a new model version, allowing real-user validation before full commitment. The risk surface is proportional to the traffic fraction sent to the canary.

```yaml
# Two Deployments sharing the same Service selector
# Ratio of replicas controls the traffic split: 9:1 = 10% canary
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-v1
  namespace: ml-serving
spec:
  replicas: 9              # 90% of traffic
  selector:
    matchLabels:
      app: llm-inference
      version: v1
  template:
    metadata:
      labels:
        app: llm-inference
        version: v1
    spec:
      containers:
        - name: server
          image: my-registry/llm-serve:1.0.0
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-v2
  namespace: ml-serving
spec:
  replicas: 1              # 10% canary traffic
  selector:
    matchLabels:
      app: llm-inference
      version: v2
  template:
    metadata:
      labels:
        app: llm-inference
        version: v2
    spec:
      containers:
        - name: server
          image: my-registry/llm-serve:1.1.0
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-svc
spec:
  selector:
    app: llm-inference    # matches BOTH v1 and v2 Pods
  ports:
    - port: 80
      targetPort: 8080
```

**Progressive rollout cadence:**

| Time | Canary % | Monitor |
|------|----------|---------|
| T+0h | 5% | Error rate, p99 latency |
| T+2h | 10% | Add: quality metric (faithfulness) |
| T+8h | 25% | Add: business metrics (task completion) |
| T+24h | 50% | A/B test for statistical significance on quality |
| T+48h | 100% | Full rollout; decommission v1 |

!!! warning "Automatic rollback triggers"
    Define rollback criteria before starting: error rate increase > 1%, p99 latency increase > 20%, or LLM quality metric drop > 5%. Use Argo Rollouts or Flagger for automated canary analysis and rollback.

---

## Exercises

1. **FastAPI endpoint extension**: Add a `/v1/embeddings` endpoint to `serve.py` that accepts a list of strings (max 100, each max 8,192 chars) and returns their embeddings from a `sentence-transformers` model. Include authentication, Pydantic validation, and a Redis-backed rate limit of 100 requests/minute per JWT token.

2. **ONNX benchmarking**: Export a fine-tuned BERT-base classifier to ONNX. Benchmark p50 and p99 latency (over 1,000 requests, batch size 1) for: (a) PyTorch eager CPU, (b) ONNX Runtime CPU, (c) ONNX Runtime with `CUDAExecutionProvider`. Report throughput (samples/sec) for each configuration.

3. **Quantisation analysis**: Load a 7B parameter LLM in BF16, INT8, and INT4 (GPTQ). Compare: peak GPU memory, throughput (tokens/sec at batch size 8), and perplexity on a held-out corpus. At what precision does perplexity increase by more than 3% relative to BF16?

4. **KV cache sizing exercise**: Llama-3-70B has 80 layers, 64 attention heads, and head dimension 128. You want to serve it in BF16 with context length 8,192 and batch size 4. Calculate: (a) KV cache memory requirement, (b) model weight memory in BF16, (c) whether this fits on two A100 80 GB GPUs, (d) whether INT4 quantisation of model weights (only) makes it feasible on one A100.

5. **Canary rollout runbook**: Write a step-by-step runbook (suitable for a junior engineer to follow at 2 AM) for a canary rollout of a new LLM version (v1 → v2) in Kubernetes. Include: pre-deployment smoke test commands, the replica ratio change sequence, monitoring commands to run at each stage, rollback commands and criteria, and the definition of "deployment success".

---

## Summary

- **Serving patterns** — online, batch, and streaming — each optimise for different latency/throughput/cost tradeoffs; the choice must be made before designing infrastructure.
- **FastAPI with vLLM** provides a production-grade async endpoint with SSE streaming, JWT auth, and Pydantic validation in under 100 lines of typed Python.
- **Model serialisation** formats (ONNX, TorchScript, SafeTensors) trade portability against performance; SafeTensors is now the safe default for LLM weights on the Hugging Face Hub.
- **Inference optimisation** — quantisation (INT8, INT4/GPTQ), continuous batching, Flash Attention, and KV cache management — can achieve 4–24× cost reduction with careful tuning.
- **vLLM's PagedAttention** eliminates KV cache fragmentation and is the current performance standard for open-model LLM serving.
- **Blue-green** deployments enable instant atomic rollback; **canary** deployments enable risk-limited progressive validation with real users; both are essential for production model lifecycle management.

*Next: [Ch 3 — ML Monitoring & Observability](../ch03-monitoring/index.md)*
