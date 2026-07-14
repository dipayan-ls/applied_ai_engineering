# Ch 1 — LLM Foundations

!!! abstract
    This chapter builds the conceptual and mathematical bedrock for all subsequent LLM engineering work.
    We trace the path from the autoregressive probability chain rule through tokenisation, the GPT
    decoder-only architecture, scaling laws, and production inference mechanics such as the KV cache
    and sampling strategies.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. State the autoregressive factorisation and explain why it is tractable to optimise.
2. Walk through the BPE algorithm on a concrete example and compare the three major tokenisation schemes.
3. Describe the GPT decoder-only architecture, including causal masking and the pre-training objective.
4. Apply Chinchilla scaling law formulas to estimate the compute-optimal dataset size for a given model.
5. Explain the KV cache and calculate its memory footprint for a given model and sequence length.

---

## 1.1 What Is a Language Model?

A language model assigns a probability to every sequence of tokens. Given a vocabulary
\(\mathcal{V}\) and a sequence of \(n\) tokens \((w_1, w_2, \ldots, w_n)\), the **autoregressive
factorisation** uses the chain rule of probability:

$$
P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, w_2, \ldots, w_{i-1})
$$

This is exact — no independence assumptions are made. The model is trained to predict each next token
given all preceding tokens, using cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(w_i \mid w_1, \ldots, w_{i-1})
$$

where \(\theta\) denotes all model parameters. Minimising this loss is equivalent to maximising the
likelihood of the training corpus, which drives the model to internalise grammar, facts, and reasoning
patterns simultaneously.

!!! note "Why autoregressive and not masked?"
    Masked language models (BERT) predict randomly masked tokens and excel at representation tasks.
    Autoregressive models predict every token left-to-right and are natural text generators. GPT-style
    models use the autoregressive approach; encoder-only models use masking. Encoder-decoder models
    (T5, BART) combine both.

---

## 1.2 Tokenisation

Raw text must be converted to integers before it can be fed into a neural network. The mapping from
text to integers is called **tokenisation**.

### 1.2.1 Byte-Pair Encoding (BPE)

BPE starts from a character vocabulary and iteratively merges the most frequent adjacent pair.

**Algorithm — BPE step-by-step**

```
Input: corpus as a list of words with end-of-word marker </w>
       e.g. {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6}

1. Initialise vocabulary V = {all unique characters + </w>}
2. Repeat num_merges times:
   a. Count all adjacent symbol pairs across the corpus.
   b. Find the most frequent pair (e.g. "e s").
   c. Merge that pair into a single symbol "es".
   d. Replace all occurrences of "e s" with "es" in the corpus.
   e. Add "es" to V.
3. Return V and the ordered merge table.
```

At inference time, the same merge table is applied greedily in order to tokenise unseen text.

**Concrete trace (first two merges)**

| Step | Most frequent pair | New symbol | Corpus state |
|------|--------------------|------------|--------------|
| Init | — | — | `l o w </w>`, `n e w e s t </w>` |
| 1 | `(e, s)` → freq 6 | `es` | `l o w </w>`, `n e w es t </w>` |
| 2 | `(es, t)` → freq 6 | `est` | `l o w </w>`, `n e w est </w>` |

### 1.2.2 WordPiece

WordPiece, used in BERT, selects the merge that maximises:

$$
\text{score}(A, B) = \frac{P(AB)}{P(A) \cdot P(B)}
$$

This prefers merges that are surprising relative to the individual token frequencies, producing
linguistically motivated subwords (e.g. `un##happy`). Continuations are marked with `##`.

### 1.2.3 SentencePiece

SentencePiece treats whitespace as a regular character (using `▁` as a stand-in) and trains
directly on raw Unicode bytes without a pre-tokenisation step. This makes it language-agnostic and
robust to languages without whitespace delimiters (Chinese, Japanese, Thai).

### 1.2.4 Tokenisation Scheme Comparison

| Property | BPE | WordPiece | SentencePiece |
|----------|-----|-----------|---------------|
| Merge criterion | Frequency | Likelihood ratio | Frequency (BPE/Unigram LM) |
| Language dependency | Requires whitespace split | Requires whitespace split | Language-agnostic |
| Continuation marker | None (context-implied) | `##` prefix | `▁` space prefix |
| Typical models | GPT-2, GPT-4, LLaMA | BERT, DistilBERT | T5, Gemma, LLaMA-3 |
| Vocabulary size | 32 k – 100 k | 30 k | 32 k – 256 k |

### 1.2.5 Token vs Word vs Character

| Unit | English vocab size | Pros | Cons |
|------|--------------------|------|------|
| Character | ~256 (bytes) | No OOV problem | Very long sequences; limited semantics per token |
| Word | 100 k – 1 M | Interpretable | Huge vocabulary; OOV for morphological variants |
| Subword (BPE) | 32 k – 100 k | Balance of both; handles rare words | Splits can be counterintuitive |

A larger vocabulary means shorter sequences (fewer tokens per sentence) and lower memory for the
sequence, but a larger embedding matrix. GPT-4 uses ~100 k tokens; LLaMA-3 uses 128 k.

---

## 1.3 The GPT Architecture

GPT models use a **decoder-only transformer**: a stack of transformer blocks where each token attends
only to itself and all preceding tokens.

### 1.3.1 Architectural Overview

```mermaid
graph LR
    IN["Token IDs\n(batch × seq_len)"] --> E["Embedding\n+ Positional Encoding"]
    E --> B1["Transformer Block 1\n(Masked Self-Attn + FFN)"]
    B1 --> B2["Transformer Block 2"]
    B2 --> BN["... Block N"]
    BN --> LN["Layer Norm"]
    LN --> LMH["LM Head\n(Linear → vocab_size)"]
    LMH --> P["Next-token\nProbabilities"]

    style IN fill:#343a40,color:#fff
    style P fill:#0d6efd,color:#fff
```

### 1.3.2 Causal Masking

Standard self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

In a causal (decoder-only) model, a lower-triangular mask is applied before the softmax so that
position \(i\) cannot attend to positions \(j > i\):

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

This allows all positions to be processed in parallel during training while preserving the
autoregressive property.

### 1.3.3 Pre-training Objective

During pre-training the model receives a long sequence of text and is trained to predict every next
token. Because each position predicts the next token, one forward pass through a sequence of length
\(L\) yields \(L-1\) training signals — making GPT-style training extremely data-efficient relative
to the number of forward passes.

---

## 1.4 Scaling Laws

### 1.4.1 Kaplan et al. 2020

Kaplan and colleagues at OpenAI empirically measured how loss scales with three independent variables:

$$
L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}
$$

where \(N\) = parameters, \(D\) = training tokens, \(C\) = compute (FLOPs), and
\(\alpha_N \approx 0.076\), \(\alpha_D \approx 0.095\).

Their key finding: **model size matters more than dataset size** when compute is fixed. This led to
the practice of training very large models on relatively small data budgets.

### 1.4.2 Chinchilla (Hoffmann et al. 2022)

DeepMind's Chinchilla paper reran the scaling study with more controlled compute budgets and found
that the Kaplan exponents were incorrect because previous experiments were **undertrained**. The
Chinchilla result:

$$
N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

This means the optimal number of training tokens \(D\) should scale **linearly** with the number of
parameters \(N\). The rule-of-thumb:

$$
D_{\text{opt}} \approx 20 \times N
$$

!!! example "Chinchilla sizing example"
    A 7 B parameter model should be trained on approximately \(20 \times 7 \times 10^9 = 140\) billion
    tokens for compute-optimal training. LLaMA-2-7B was trained on 2 T tokens — well beyond Chinchilla
    optimal, which yields a higher-quality model at inference time at the cost of more training compute.

### 1.4.3 Beyond Chinchilla

Later work (LLaMA, Mistral) showed that over-training relative to Chinchilla is rational when
**inference compute is the bottleneck**: a smaller model trained longer is cheaper to serve than
a larger model trained to the same loss.

---

## 1.5 Pre-training Data

### 1.5.1 Data Mixture

Modern LLMs are trained on a mixture of sources weighted by quality:

| Source | Typical weight | Notes |
|--------|---------------|-------|
| Web (Common Crawl) | 40 – 70 % | Largest volume; noisy |
| Books | 10 – 20 % | Long-range coherence |
| Code | 5 – 20 % | Improves reasoning |
| Wikipedia | 2 – 5 % | High-quality factual text |
| Scientific papers | 2 – 5 % | Domain depth |
| Curated web (C4, RefinedWeb) | 10 – 30 % | Filtered Common Crawl |

### 1.5.2 Data Cleaning

Raw web data contains boilerplate, HTML artefacts, adult content, and spam. A standard cleaning
pipeline includes:

1. Language identification (fastText LangID).
2. URL and domain blocklisting.
3. Heuristic filtering (line length, punctuation ratio, symbol fraction).
4. Perplexity filtering with a small n-gram language model.
5. Exact and near-duplicate removal (MinHash, suffix arrays).

!!! warning "Deduplication is critical"
    Models memorise repeated sequences verbatim. Deduplicated training sets produce lower loss,
    better generalisation, and lower risk of copyright-infringing verbatim recitation.
    Lee et al. (2022) showed that deduplication improves benchmark performance by ~1–3 %.

---

## 1.6 Positional Embeddings

Transformers have no built-in notion of sequence order. Positional information is injected either
before the first layer or inside each attention operation.

### 1.6.1 Absolute Positional Embeddings

The original transformer adds fixed sinusoidal vectors to token embeddings:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

GPT-2 uses learned absolute embeddings. Both approaches struggle to generalise beyond the training
context length.

### 1.6.2 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) encodes position by rotating query and key vectors in the complex plane before
the dot-product:

$$
\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \cdot e^{im\theta}
$$

The dot product then naturally captures relative distances:

$$
\langle \text{RoPE}(\mathbf{q}, m),\, \text{RoPE}(\mathbf{k}, n) \rangle = f(m - n)
$$

RoPE can be extrapolated beyond training length with YaRN or NTK-aware scaling. Used in LLaMA,
Mistral, Gemma, and GPT-NeoX families.

### 1.6.3 ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) adds a fixed negative linear bias proportional to the distance between
query and key positions, directly to the attention logits:

$$
\text{Attention}_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}} - m \cdot |i - j|
$$

where \(m\) is a head-specific slope. ALiBi generalises strongly beyond the training length without
any fine-tuning. Used in MPT and BLOOM.

### 1.6.4 Positional Embedding Comparison

| Scheme | Relative position | Extrapolation | Memory cost | Models |
|--------|-------------------|---------------|-------------|--------|
| Sinusoidal absolute | No | Poor | None | Original Transformer |
| Learned absolute | No | Poor | `seq_len × d` | GPT-2 |
| RoPE | Yes | Good (with scaling) | None | LLaMA, Gemma, Mistral |
| ALiBi | Yes | Excellent | None | MPT, BLOOM |

---

## 1.7 The KV Cache

During **autoregressive inference** the model generates one token at a time. At step \(t\), it
computes keys and values for all \(t\) tokens. Without caching, this is \(O(t^2)\) per token
generated.

The **KV cache** stores the key and value tensors from all previous positions. At step \(t\) only
the new token's queries are computed; cached keys and values handle all prior context.

**Memory cost of the KV cache:**

$$
\text{Mem}_{KV} = 2 \times \text{num\_layers} \times \text{num\_heads} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes\_per\_element}
$$

!!! example "KV cache for LLaMA-3-8B"
    LLaMA-3-8B has 32 layers, 8 KV heads (GQA), head dimension 128, bfloat16 (2 bytes).
    For a sequence of 8 192 tokens:

    $$
    2 \times 32 \times 8 \times 128 \times 8192 \times 2 = 1.07 \text{ GB}
    $$

    With grouped-query attention (GQA) the cache size scales with KV heads (8), not all attention
    heads (32), giving a 4× saving versus multi-head attention.

---

## 1.8 Sampling Strategies

After the LM head produces logits \(\mathbf{z} \in \mathbb{R}^{|\mathcal{V}|}\), a sampling
strategy converts them into the next token.

### 1.8.1 Greedy Decoding

$$
w_t = \arg\max_w P(w \mid w_{<t})
$$

Deterministic. Tends to produce repetitive text; can get stuck in loops.

### 1.8.2 Temperature Scaling

Temperature \(T\) reshapes the distribution before sampling:

$$
P_T(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- \(T \to 0\): approaches greedy (sharp peak).
- \(T = 1\): raw model distribution.
- \(T > 1\): flatter distribution, more diverse but less coherent.

### 1.8.3 Top-k Sampling

Keep only the top \(k\) most probable tokens and re-normalise:

$$
P_{\text{top-}k}(w_i) = \frac{P(w_i) \cdot \mathbf{1}[w_i \in \text{top-}k]}{\sum_{j \in \text{top-}k} P(w_j)}
$$

Typical value: \(k = 50\). Prevents sampling from very low-probability tokens but the cutoff is
fixed regardless of how peaked or flat the distribution is.

### 1.8.4 Nucleus (Top-p) Sampling

Sort tokens by probability descending; include the smallest set \(\mathcal{V}_p\) whose cumulative
probability exceeds \(p\):

$$
\mathcal{V}_p = \arg\min_{\mathcal{V}' \subseteq \mathcal{V}} \left\{ |\mathcal{V}'| \;\Big|\; \sum_{w \in \mathcal{V}'} P(w) \geq p \right\}
$$

Then sample uniformly from \(\mathcal{V}_p\) after re-normalisation. Typical value: \(p = 0.9\) or
\(0.95\). Adapts the vocabulary size to the uncertainty of the distribution.

### 1.8.5 Sampling Strategy Comparison

| Strategy | Deterministic | Controls diversity | Truncation |
|----------|--------------|-------------------|------------|
| Greedy | Yes | No | No |
| Temperature | No | Via \(T\) | No |
| Top-k | No | Via \(k\) | Hard cutoff at rank \(k\) |
| Top-p (nucleus) | No | Via \(p\) | Adaptive to distribution |

!!! tip "Practical defaults"
    Most production chat APIs use temperature 0.7 + top-p 0.9 as defaults. For code generation,
    temperature 0 (greedy) or 0.2 with top-p 0.95 is common. Combining top-k and top-p is
    generally redundant — choose one.

---

## 1.9 Key Model Families

| Model family | Organisation | Architecture | Context length | Open weights | Key innovation |
|-------------|-------------|-------------|----------------|-------------|----------------|
| GPT-4 | OpenAI | Decoder-only, MoE (rumoured) | 128 k | No | Scale; multimodal |
| LLaMA 3 | Meta | Decoder-only, GQA, RoPE | 128 k | Yes | Open frontier-class model |
| Mistral 7B | Mistral AI | Decoder-only, GQA, sliding window | 32 k | Yes | SWA; strong at 7 B |
| Mixtral 8×7B | Mistral AI | Sparse MoE (2 of 8 experts) | 32 k | Yes | MoE efficiency |
| Gemma 2 | Google DeepMind | Decoder-only, GQA, logit soft-cap | 8 k | Yes | Knowledge distillation |
| Claude 3.5 | Anthropic | Decoder-only (details NDA) | 200 k | No | Long context; safety |
| Phi-3 | Microsoft | Decoder-only | 128 k | Yes | High quality at small scale |

!!! note "GQA — Grouped-Query Attention"
    Most modern open models use GQA instead of multi-head attention. GQA shares key-value heads
    across groups of query heads, reducing KV cache size by a factor of `num_heads / num_kv_heads`
    without measurable quality loss.

---

## 1.10 Summary

- Language models factorise the joint token probability autoregressively; cross-entropy training maximises
  log-likelihood of next-token predictions.
- BPE, WordPiece, and SentencePiece all perform subword segmentation but differ in merge criterion and
  language assumptions. Modern frontier models use vocabulary sizes of 32 k – 128 k.
- GPT uses a decoder-only transformer with causal masking, enabling parallel training while preserving the
  left-to-right generation property.
- Chinchilla scaling laws prescribe ~20 training tokens per parameter; over-training is justified when
  inference cost matters more than training cost.
- The KV cache eliminates redundant recomputation during autoregressive inference; GQA shrinks its memory
  footprint 4 – 8×.
- Top-p (nucleus) sampling provides adaptive vocabulary truncation and is the most commonly used
  production sampling strategy.

---

## Exercises

1. **Tokenisation practice.** Using the `tiktoken` library, tokenise the sentence
   *"The Mixtral 8×7B model uses a sparse mixture-of-experts architecture."* with the `cl100k_base`
   encoding. How many tokens are produced? Which tokens surprise you?

2. **Scaling law prediction.** Assume a compute budget of \(10^{23}\) FLOPs and a typical transformer
   where each forward pass costs \(\approx 6N\) FLOPs per token. Using Chinchilla, estimate the
   optimal number of parameters \(N\) and dataset tokens \(D\).

3. **KV cache sizing.** Calculate the KV cache memory for a 70 B parameter LLaMA-3 model (80 layers,
   8 KV heads, head dim 128, bfloat16) at a context length of 32 k tokens. How does this compare to
   the model weights?

4. **Sampling exploration.** Using the Anthropic API, call `claude-3-5-haiku-20241022` with
   temperatures 0, 0.5, 1.0, and 1.5 on the prompt *"List three potential names for a new coffee
   shop."* Run each temperature five times. Describe the observed diversity and coherence tradeoffs.

5. **RoPE vs ALiBi.** Both RoPE and ALiBi encode relative positions without adding to the parameter
   count. Explain in your own words why ALiBi extrapolates more reliably to lengths unseen during
   training than vanilla RoPE (without NTK scaling). What is the intuitive geometric difference?
