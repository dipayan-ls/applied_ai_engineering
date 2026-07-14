# Roadmap

## Curriculum Map

```mermaid
graph TD
    V1["Vol 1<br/>Foundations"] --> V2["Vol 2<br/>Python Engineering"]
    V2 --> V3["Vol 3<br/>Machine Learning"]
    V3 --> V4["Vol 4<br/>Deep Learning"]
    V4 --> V5["Vol 5<br/>Transformers"]
    V5 --> V6["Vol 6<br/>LLMs"]
    V6 --> V7["Vol 7<br/>RAG"]
    V6 --> V8["Vol 8<br/>AI Agents"]
    V7 --> V8
    V8 --> V9["Vol 9<br/>MLOps"]
    V9 --> V10["Vol 10<br/>Enterprise AI"]

    style V1 fill:#1e40af,color:#fff
    style V2 fill:#1e40af,color:#fff
    style V3 fill:#1d4ed8,color:#fff
    style V4 fill:#1d4ed8,color:#fff
    style V5 fill:#2563eb,color:#fff
    style V6 fill:#2563eb,color:#fff
    style V7 fill:#3b82f6,color:#fff
    style V8 fill:#3b82f6,color:#fff
    style V9 fill:#60a5fa,color:#fff
    style V10 fill:#93c5fd,color:#000
```

## Completion Status

| Volume | Title | Chapters | Status |
|---|---|---|---|
| 01 | Foundations of AI | Ch 0-3 | In Progress |
| 02 | Python Engineering | Ch 1-3 | In Progress |
| 03 | Machine Learning | Ch 1-4 | Planned |
| 04 | Deep Learning | Ch 1-4 | Planned |
| 05 | Transformers | Ch 1-3 | Planned |
| 06 | Large Language Models | Ch 1-4 | Planned |
| 07 | RAG | Ch 1-3 | Planned |
| 08 | AI Agents | Ch 1-3 | Planned |
| 09 | MLOps | Ch 1-3 | Planned |
| 10 | Enterprise AI | Ch 1-3 | Planned |

## Modern AI Stack

```mermaid
graph TB
    User["User / Application"]
    FE["Frontend"]
    API["API Gateway"]
    BL["Business Logic"]
    Orch["LLM Orchestration<br/>(LangChain / LlamaIndex)"]
    Ret["Retrieval Layer"]
    VDB["Vector Database"]
    LLM["LLM Inference"]
    Mon["Observability & Monitoring"]

    User --> FE --> API --> BL --> Orch
    Orch --> Ret --> VDB
    Orch --> LLM
    LLM --> Mon
    Orch --> Mon

    style LLM fill:#7c3aed,color:#fff
    style VDB fill:#0891b2,color:#fff
    style Mon fill:#059669,color:#fff
```

!!! note
    LLMs are **one component** of a production AI system. This course covers the entire stack.
