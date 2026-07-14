# Ch 2 — Agent Frameworks

> **Volume 8 · Chapter 2** | Estimated reading time: 65 minutes

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Compare the five major agent frameworks — LangChain, LlamaIndex, AutoGen, CrewAI, and the Anthropic Agents SDK — on the dimensions of maturity, abstraction level, multi-agent support, observability, and community size.
2. Build a working LangChain LCEL pipeline that chains retrieval, prompt templating, and LLM invocation.
3. Implement a document Q&A agent using LlamaIndex query engines with sub-question decomposition.
4. Configure LangSmith or Langfuse tracing to capture agent traces and identify bottlenecks in production.
5. Design production-ready agents with retry logic, fallbacks, rate-limit handling, and per-request cost monitoring.

---

## 1. Framework Comparison

Choosing an agent framework is a consequential decision: it determines your abstraction boundaries, debugging surface area, and migration cost. The table below summarises the five frameworks practitioners encounter most frequently.

| Framework | Maturity | Abstraction | Multi-Agent | Observability | Community |
|-----------|----------|-------------|-------------|---------------|-----------|
| **LangChain** | High (v0.3+) | High — many built-in chains and agents | Moderate (LangGraph) | Excellent (LangSmith) | Very large |
| **LlamaIndex** | High (v0.10+) | Medium — focused on data ingestion and query | Moderate (agent workflows) | Good (Langfuse, Arize) | Large |
| **AutoGen** | Medium | Low — close-to-metal multi-agent messaging | Excellent (native) | Moderate | Growing |
| **CrewAI** | Medium | High — role-based crew abstraction | Excellent (native) | Moderate | Growing |
| **Anthropic Agents SDK** | Early | Low — thin wrapper around Claude tool_use | Early | Via Langfuse | Small |

!!! tip "When to choose which"
    - **LangChain**: large teams, broad ecosystem integrations, LangSmith for observability.
    - **LlamaIndex**: document-heavy applications where the data ingestion pipeline is the primary concern.
    - **AutoGen**: research on conversational multi-agent systems; requires more glue code.
    - **CrewAI**: rapid prototyping of role-based agent teams.
    - **Anthropic SDK**: when you want the thinnest abstraction over Claude's native tool_use API.

---

## 2. LangChain

LangChain provides composable primitives for LLM applications. The core abstractions are **Chains**, **Agents**, **Tools**, and (in v0.3+) the **LangChain Expression Language (LCEL)**.

### 2.1 Chains

A **chain** is a sequence of LLM calls or other operations. In LCEL, chains are expressed using the `|` (pipe) operator:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise technical writer."),
        ("human", "Explain {topic} in exactly two sentences."),
    ]
)

chain = prompt | llm | StrOutputParser()

result: str = chain.invoke({"topic": "vector databases"})
print(result)
```

### 2.2 Agents and Tools

LangChain agents wrap an LLM with tools and a looping mechanism. The `create_tool_calling_agent` function works with any LLM that supports native tool calling:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-opus-4-5")
tools = [DuckDuckGoSearchRun()]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant with access to web search."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)

result = executor.invoke({"input": "What is the population of Tokyo as of 2024?"})
print(result["output"])
```

### 2.3 LangChain Expression Language (LCEL)

LCEL is LangChain's declarative composition layer. It provides:

- **Streaming**: `chain.stream(input)` yields tokens as they are generated.
- **Async**: `await chain.ainvoke(input)` for non-blocking calls.
- **Batching**: `chain.batch([input1, input2])` runs in parallel.
- **Fallbacks**: `chain.with_fallbacks([backup_chain])`.

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # fallback provider
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

primary_llm = ChatAnthropic(model="claude-opus-4-5")
fallback_llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("Summarise: {text}")
parser = StrOutputParser()

# If the primary LLM raises an exception, fall back to OpenAI
chain = (prompt | primary_llm.with_fallbacks([fallback_llm]) | parser)

summary = chain.invoke({"text": "LangChain is a framework for building LLM applications..."})
```

---

## 3. LlamaIndex

LlamaIndex (formerly GPT Index) specialises in connecting LLMs to enterprise data. Its core abstraction is the **Query Engine**: a component that accepts a natural-language question, retrieves relevant documents, and synthesises an answer.

### 3.1 Query Engines

```
User question → Retriever → Re-ranker → Synthesiser → Answer
```

LlamaIndex ships with over a dozen retriever types: vector, keyword, knowledge graph, SQL, and composites.

### 3.2 Routers

A **Router** sends a query to the most appropriate query engine based on metadata:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

financial_engine = ...  # query engine over financial documents
technical_engine = ...  # query engine over API docs

tools = [
    QueryEngineTool.from_defaults(
        query_engine=financial_engine,
        description="Use for questions about revenue, costs, and financial metrics.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=technical_engine,
        description="Use for questions about APIs, code, and technical specifications.",
    ),
]

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
)
```

### 3.3 Sub-Question Decomposition

For complex multi-part questions, LlamaIndex can break the question into sub-questions, run each against the appropriate engine, and synthesise the results:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    verbose=True,
)

response = sub_question_engine.query(
    "Compare Q3 revenue and explain the API rate-limit changes introduced in the same quarter."
)
```

---

## 4. Building a Document Q&A Agent with LlamaIndex

The following is a complete, runnable document Q&A agent. It ingests a directory of PDF files, builds a vector index, wraps it in a ReAct agent, and exposes a query interface.

```python
"""
doc_qa_agent.py — LlamaIndex document Q&A agent with ReAct loop.

Requirements:
    pip install llama-index>=0.10 llama-index-llms-anthropic \
                llama-index-embeddings-openai pypdf
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic


def build_document_agent(docs_dir: str | Path) -> ReActAgent:
    """
    Ingest documents from `docs_dir`, build a vector index,
    and wrap it in a LlamaIndex ReActAgent.

    Args:
        docs_dir: Directory containing PDF, TXT, or Markdown files.

    Returns:
        A ReActAgent ready to answer questions about the documents.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_path}")

    # Configure global LLM and embedding model
    Settings.llm = Anthropic(
        model="claude-opus-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=4096,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Load and index documents
    print(f"Loading documents from {docs_path} ...")
    documents = SimpleDirectoryReader(
        input_dir=str(docs_path),
        recursive=True,
        required_exts=[".pdf", ".txt", ".md"],
    ).load_data()
    print(f"Loaded {len(documents)} document chunks.")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = index.as_query_engine(similarity_top_k=6)

    # Wrap query engine as an agent tool
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="document_search",
            description=(
                "Search the document corpus and retrieve relevant information. "
                "Input should be a precise natural-language question."
            ),
        ),
    )

    agent = ReActAgent.from_tools(
        tools=[query_tool],
        llm=Settings.llm,
        verbose=True,
        max_iterations=10,
        context=(
            "You are a knowledgeable assistant that answers questions "
            "strictly based on the provided documents. "
            "If the answer is not in the documents, say so clearly."
        ),
    )

    return agent


def main() -> None:
    docs_directory = os.environ.get("DOCS_DIR", "./documents")
    agent = build_document_agent(docs_directory)

    print("\nDocument Q&A Agent ready. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        response = agent.chat(question)
        print(f"\nAnswer: {response}\n")


if __name__ == "__main__":
    main()
```

### 4.1 Persistence and Scalability

For production deployments, persist the index to disk or a vector database so it does not have to be rebuilt on every start:

```python
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

INDEX_PERSIST_DIR = "./index_storage"

def save_index(index: VectorStoreIndex) -> None:
    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)

def load_index() -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_PERSIST_DIR
    )
    return load_index_from_storage(storage_context)
```

---

## 5. Anthropic's Claude API for Agentic Tasks

When you need minimal abstraction and maximum control, use the Anthropic Python SDK directly. Claude's native `tool_use` content type gives you a clean, framework-agnostic agent loop.

### 5.1 The `tool_use` Content Type

When the model decides to call a tool, it returns a content block of type `tool_use`:

```json
{
  "type": "tool_use",
  "id": "toolu_01XFDUDYJgAACTU9zNBT4TZ6",
  "name": "search_web",
  "input": {"query": "Tokyo population 2024"}
}
```

The agent must execute the tool and return a `tool_result` block in the next user turn:

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01XFDUDYJgAACTU9zNBT4TZ6",
  "content": "Tokyo's population as of 2024 is approximately 13.96 million..."
}
```

### 5.2 Streaming Agent Loops

For real-time user feedback, use the streaming API inside the agent loop:

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def stream_agent_step(
    messages: list[dict],
    tools: list[dict],
    system: str,
) -> tuple[list, str]:
    """
    Run one streaming step of the agent loop.

    Returns:
        (content_blocks, stop_reason)
    """
    content_blocks = []
    stop_reason = ""

    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system,
        tools=tools,
        messages=messages,
    ) as stream:
        for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        print(event.delta.text, end="", flush=True)

        final_message = stream.get_final_message()
        content_blocks = final_message.content
        stop_reason = final_message.stop_reason

    print()  # newline after streamed text
    return content_blocks, stop_reason
```

---

## 6. Observability: Tracing Agent Runs

In production, a single agent run may involve 10–20 LLM calls. Without distributed tracing, debugging is guesswork.

### 6.1 LangSmith

LangSmith is LangChain's observability platform. It captures every LLM call, tool invocation, and chain execution automatically when the environment variable `LANGCHAIN_TRACING_V2=true` is set.

```bash
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-agent-project"
```

No code changes are required for LangChain-based agents. The trace appears in the LangSmith dashboard with full input/output, token counts, and latency for each step.

### 6.2 Langfuse

Langfuse is an open-source alternative that works with any LLM framework. It provides a Python SDK for manual instrumentation:

```python
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
import os

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


@observe(name="agent_run")
def traced_agent_run(user_message: str) -> str:
    """Agent run decorated with Langfuse tracing."""
    langfuse_context.update_current_trace(
        name="ReAct Agent",
        user_id="user-123",
        tags=["production", "v2"],
    )

    # ... your agent loop here ...
    answer = run_agent(user_message)

    langfuse_context.update_current_observation(
        output=answer,
        metadata={"model": "claude-opus-4-5"},
    )
    return answer
```

### 6.3 What to Trace

| Signal | Why it matters |
|--------|---------------|
| LLM call latency (per step) | Identifies slow prompts or model cold starts |
| Input/output token counts | Cost attribution; context budget monitoring |
| Tool call name and arguments | Debugging wrong tool selection |
| Tool call duration | Identifies slow external APIs |
| Agent step count | Detects runaway agents |
| Final answer quality score | Closed-loop quality monitoring |

---

## 7. Prompt Management in Agentic Systems

Unlike single-turn applications, agents use multiple prompts across multiple steps. Prompt management becomes a first-class concern.

### 7.1 Centralised Prompt Registry

Store prompts as versioned templates in a dedicated registry (Langfuse Prompt Management, or a simple database). Reference them by name and version:

```python
# Fetch a prompt from Langfuse prompt management
prompt_template = langfuse.get_prompt("react-agent-system", version=3)
system_prompt: str = prompt_template.compile(agent_name="Research Assistant")
```

### 7.2 Prompt Versioning

- Tag each prompt version with a semantic version (e.g., `v1.2.0`).
- Pin production agents to a specific prompt version to avoid unexpected behaviour changes.
- A/B test new prompt versions against a shadow cohort before rolling out.

### 7.3 Dynamic Prompt Construction

System prompts often include context that varies per request: user role, available tools, retrieved knowledge. Use a template engine (Jinja2) rather than string concatenation:

```python
from jinja2 import Template

SYSTEM_TEMPLATE = Template(
    """\
You are {{ agent_name }}, an AI assistant with the following tools:
{% for tool in tools %}
- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}

Current date: {{ current_date }}
User tier: {{ user_tier }}
"""
)

system_prompt = SYSTEM_TEMPLATE.render(
    agent_name="ResearchBot",
    tools=TOOL_SCHEMAS,
    current_date="2025-07-15",
    user_tier="enterprise",
)
```

---

## 8. Production Considerations

### 8.1 Retries and Fallbacks

LLM APIs are not perfectly reliable. Implement exponential backoff with jitter:

```python
import time
import random
import anthropic

client = anthropic.Anthropic()


def call_with_retry(
    messages: list[dict],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> anthropic.types.Message:
    """Call the Anthropic API with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                messages=messages,
            )
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Rate limited. Retrying in {delay:.1f}s ...")
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:  # Server error — retry
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise  # Client error — don't retry
    raise RuntimeError("Unreachable")
```

### 8.2 Rate Limiting

Track tokens-per-minute (TPM) and requests-per-minute (RPM) against your API tier limits. Use a token bucket or sliding window counter:

```python
import threading
import time
from collections import deque


class RateLimiter:
    """Sliding window rate limiter for API calls."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            # Evict timestamps outside the window
            while self._timestamps and self._timestamps[0] < now - self.window:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_requests:
                sleep_for = self.window - (now - self._timestamps[0])
                time.sleep(max(sleep_for, 0))
            self._timestamps.append(time.monotonic())


# Example: 60 requests per minute
limiter = RateLimiter(max_requests=60, window_seconds=60.0)
```

### 8.3 Cost Monitoring

Track per-request costs and alert when they exceed a threshold. Anthropic's API returns `usage` with `input_tokens` and `output_tokens`:

```python
COST_PER_MILLION_INPUT = 3.00   # USD for claude-opus-4-5
COST_PER_MILLION_OUTPUT = 15.00

def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Return the cost in USD for a single API call."""
    return (
        input_tokens * COST_PER_MILLION_INPUT / 1_000_000
        + output_tokens * COST_PER_MILLION_OUTPUT / 1_000_000
    )

# After each API call:
# response.usage.input_tokens, response.usage.output_tokens
```

---

## 9. Exercises

1. **LCEL pipeline**: Build a LangChain LCEL chain that (a) retrieves the top-3 documents from a Chroma vector store, (b) formats them into a context block, (c) runs a Claude completion to answer a question, and (d) streams the output token-by-token to stdout.

2. **Framework benchmark**: Run the same document Q&A task (10 questions from a PDF of your choice) against both the LangChain and LlamaIndex implementations. Compare: mean latency per question, total tokens consumed, and subjective answer quality. Summarise findings in a table.

3. **Langfuse integration**: Instrument the `doc_qa_agent.py` from Section 4 with Langfuse traces. Add a custom score of 0 or 1 for each answer based on whether it contains a citation to a source document.

4. **Prompt A/B test**: Create two versions of the system prompt for the ReAct agent from Chapter 1. Use Langfuse to track which version produces correct answers more often over 50 test queries.

5. **Cost budget**: Modify the agent loop from Chapter 1 to abort if the cumulative cost for a single task exceeds $0.05. Return a user-friendly message explaining that the budget was exceeded.

---

## Summary

- **Framework selection** is a trade-off between abstraction level, multi-agent support, ecosystem breadth, and observability quality.
- **LangChain** excels for applications needing broad integrations and full-featured observability via LangSmith; LCEL provides composable, streaming-native pipelines.
- **LlamaIndex** is purpose-built for data-intensive applications; its router and sub-question engine handle complex document corpora elegantly.
- **Anthropic's native API** is the right choice when you need maximum control and minimal overhead; the `tool_use` / `tool_result` message protocol is the foundation all higher-level frameworks build on.
- **Observability** via LangSmith or Langfuse is non-negotiable in production — without it, debugging multi-step agent failures is prohibitively difficult.
- **Production readiness** requires: exponential backoff retries, sliding-window rate limiting, per-request cost tracking, and versioned prompt management.

---

*Next: [Ch 3 — Multi-Agent Systems](../ch03-multi-agent/index.md)*
