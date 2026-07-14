---
title: "Ch 2 — The Transformer Architecture"
---

# Ch 2 — The Transformer Architecture

<div class="chapter-meta" markdown>
| | |
|---|---|
| **Difficulty** | Advanced |
| **Reading time** | 120 min |
| **Prerequisites** | Ch 1 (Attention Mechanism), Vol 4 Ch 2 (Training Deep Networks) |
</div>

---

## Learning Objectives

By the end of this chapter you will be able to:

1. **Explain** why positional encoding is necessary for Transformers, compare sinusoidal, learned, and RoPE encodings, and implement sinusoidal encoding from the closed-form formula.
2. **Describe** the structure of a Transformer encoder block (self-attention + FFN + two Add&Norm sublayers) and the decoder block (masked self-attention + cross-attention + FFN + three Add&Norm sublayers).
3. **Distinguish** Pre-LN from Post-LN placement of layer normalisation and explain why modern large language models prefer Pre-LN for training stability.
4. **Implement** a complete Transformer encoder-decoder in PyTorch (~120 lines) with all components correctly assembled and train it on a toy sequence-to-sequence task using teacher forcing and label-smoothed cross-entropy.
5. **Select** appropriate hyperparameters (model dimension, heads, layers, FFN dimension, dropout) for small, base, and large Transformer configurations using the reference table and scaling intuition.

---

## 2.1 High-Level Architecture Overview

The Transformer (Vaswani et al., 2017) consists of an **encoder** and a **decoder**, each composed of a stack of $N$ identical blocks.

**Encoder**: maps a source sequence $x_1, \ldots, x_n$ into a sequence of continuous representations $z_1, \ldots, z_n$. No autoregressive constraint — every position attends to every other position simultaneously.

**Decoder**: generates the target sequence $y_1, \ldots, y_m$ one token at a time, conditioning each step on the encoder output and the generated prefix so far via causally masked self-attention.

Both halves use the same two design principles throughout:

1. **Residual connections**: every sublayer's output is $x + \text{Sublayer}(x)$, allowing gradients to flow through the addition unimpeded.
2. **Layer normalisation**: applied either after the residual (Post-LN, original paper) or before the sublayer input (Pre-LN, modern practice).

---

## 2.2 Positional Encoding

### 2.2.1 Why Position Information Is Needed

Self-attention is **permutation-equivariant**: shuffling the input tokens produces the same output, shuffled identically. Unlike RNNs, Transformers have no inherent notion of order — the word "not" at position 3 and "not" at position 17 produce identical representations unless positional information is added explicitly.

Positional encodings are added to the token embeddings before the first encoder and decoder layer:

$$\text{InputRepresentation}_i = \text{TokenEmbedding}(x_i) + \text{PositionalEncoding}(i)$$

### 2.2.2 Sinusoidal Positional Encoding

The original paper uses fixed sinusoidal functions of different wavelengths. For position $\text{pos}$ and embedding dimension index $i$:

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**Key properties**:

1. **Unique encoding**: no two positions produce the same $d_{\text{model}}$-dimensional vector.
2. **Smooth interpolation**: nearby positions produce more similar encodings than distant ones.
3. **Relative position via dot product**: $PE_{\text{pos}} \cdot PE_{\text{pos}+k}$ depends only on the offset $k$, not the absolute position — enabling the model to learn relative positional relationships from absolute encodings.
4. **Length generalisation**: sinusoidal PE is defined for any length, so a model trained at length 512 can in principle process length 1024 at inference.

=== "Python"

    ```python
    from __future__ import annotations

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor


    class SinusoidalPositionalEncoding(nn.Module):
        """
        Fixed sinusoidal positional encoding (Vaswani et al., 2017).

        Parameters
        ----------
        d_model  : Embedding (model) dimension.
        max_len  : Maximum sequence length to pre-compute encodings for.
        dropout  : Dropout probability applied after adding position encoding.
        """

        def __init__(
            self,
            d_model: int,
            max_len: int   = 5_000,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            pe  = torch.zeros(max_len, d_model)              # (max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float() # (max_len, 1)
            div = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10_000.0) / d_model)
            )   # (d_model // 2,) — the 1/10000^(2i/d) frequencies

            pe[:, 0::2] = torch.sin(pos * div)   # even dimensions: sin
            pe[:, 1::2] = torch.cos(pos * div)   # odd  dimensions: cos

            # Register as buffer: persisted in model state but not a learned parameter
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: Tensor) -> Tensor:
            """
            Add positional encoding to token embeddings.

            Parameters
            ----------
            x : (batch, seq_len, d_model) token embeddings.

            Returns
            -------
            Tensor of same shape with positional information added and dropout applied.
            """
            x = x + self.pe[:, : x.size(1), :]   # broadcast over batch
            return self.dropout(x)
    ```

### 2.2.3 Learned Positional Embeddings

BERT and GPT-2 use a simple `nn.Embedding(max_position, d_model)` initialised randomly and trained end-to-end. The model learns exactly what positional information is useful for the task.

**Advantage**: fully flexible — no assumption about sinusoidal structure.  
**Disadvantage**: cannot generalise beyond `max_position` seen during training; adds $\text{max\_len} \times d_{\text{model}}$ parameters.

### 2.2.4 Rotary Position Embedding (RoPE)

Modern LLMs — LLaMA, GPT-NeoX, Mistral, Gemma — use **RoPE** (Su et al., 2021). Rather than adding positional information to token embeddings, RoPE **rotates** the Q and K vectors by a position-dependent angle before the attention dot product:

$$\tilde{q}_m = \text{Rotate}(q_m,\, m\theta), \qquad \tilde{k}_n = \text{Rotate}(k_n,\, n\theta)$$

The inner product $\tilde{q}_m \cdot \tilde{k}_n$ then depends only on the **relative displacement** $m - n$. This makes long-context extrapolation more principled and is why RoPE-based models can be extended far beyond their training length via techniques like YaRN or LongRoPE.

!!! tip "RoPE in practice"
    RoPE is applied inside the attention computation, not at the embedding layer. See `transformers/models/llama/modeling_llama.py` for a production implementation. The key operation is applying a complex-number rotation in pairs of dimensions using `torch.cos` and `torch.sin`.

---

## 2.3 The Encoder Block

Each of the $N$ encoder layers wraps two sublayers with the **Add&Norm** pattern:

$$\text{Add\&Norm}(x, \text{Sublayer}) = \text{LayerNorm}\!\left(x + \text{Sublayer}(x)\right) \quad \text{(Post-LN)}$$

**Sublayer 1 — Multi-Head Self-Attention**:

$$\text{SA}(x) = \text{MultiHead}(x, x, x)$$

Every token attends to every other token in the same sequence. No causal mask is applied in the encoder.

**Sublayer 2 — Position-wise Feed-Forward Network**:

$$\text{FFN}(x) = \max(0,\; xW_1 + b_1)\, W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$.
- Applied identically and independently to each position — no cross-position mixing at this sublayer.
- Standard: $d_{ff} = 4 \times d_{\text{model}}$ (e.g., 2,048 for $d_{\text{model}} = 512$).
- The FFN dominates parameter count: $2 d_{\text{model}} d_{ff} = 8 d_{\text{model}}^2$ per layer vs. $4 d_{\text{model}}^2$ for attention.
- Modern variants replace ReLU with **SwiGLU** (LLaMA) or **GELU** (BERT, GPT) for improved performance.

**Full encoder block (Post-LN)**:

$$x' = \text{LayerNorm}\!\left(x + \text{MultiHead}(x, x, x)\right)$$
$$x'' = \text{LayerNorm}\!\left(x' + \text{FFN}(x')\right)$$

---

## 2.4 The Decoder Block

Each decoder layer has **three** sublayers:

1. **Masked multi-head self-attention**: the decoder attends to its own output prefix with a causal mask, preventing information leakage from future tokens.
2. **Multi-head cross-attention**: Q comes from the decoder; K and V come from the encoder output.
3. **Position-wise FFN**: identical structure to the encoder FFN.

All three sublayers use Add&Norm:

$$x'   = \text{LayerNorm}\!\left(x + \text{MaskedSelfAttn}(x)\right)$$
$$x''  = \text{LayerNorm}\!\left(x' + \text{CrossAttn}(x',\, \text{enc\_out})\right)$$
$$x''' = \text{LayerNorm}\!\left(x'' + \text{FFN}(x'')\right)$$

---

## 2.5 Pre-LN vs. Post-LN

### 2.5.1 Original Post-LN

$$\text{Post-LN}: \quad x^{l+1} = \text{LayerNorm}\!\left(x^l + \text{Sublayer}(x^l)\right)$$

At initialisation the gradient must pass through the LayerNorm in each stacked layer on its way to early layers. For deep stacks (>12 layers), this causes unstable gradient norms and requires careful learning rate warmup to avoid divergence.

### 2.5.2 Modern Pre-LN

$$\text{Pre-LN}: \quad x^{l+1} = x^l + \text{Sublayer}\!\left(\text{LayerNorm}(x^l)\right)$$

The gradient of the loss with respect to $x^l$ passes through the residual addition directly — no LayerNorm sits on the backward path between layers. This keeps gradient norms roughly constant across depth at initialisation, enabling stable training of deep models (24–96+ layers) without long warmup schedules.

A final LayerNorm is typically applied to the last layer's output before the prediction head (as in GPT-2, LLaMA).

!!! note "Practical recommendation"
    For models up to 12 layers: Post-LN with warmup often works. For 24+ layers: use Pre-LN. Both GPT-2 and LLaMA adopt Pre-LN; the original Vaswani et al. paper uses Post-LN.

---

## 2.6 Full Transformer PyTorch Implementation

=== "Python"

    ```python
    from __future__ import annotations

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    class SinusoidalPositionalEncoding(nn.Module):
        """Fixed sinusoidal PE (see Section 2.2.2 for full docstring)."""

        def __init__(self, d_model: int, max_len: int = 5_000, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(x + self.pe[:, : x.size(1)])


    class MultiHeadAttention(nn.Module):
        """Scaled dot-product MHA (full implementation from Chapter 1)."""

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k     = d_model // n_heads
            self.W_q     = nn.Linear(d_model, d_model)
            self.W_k     = nn.Linear(d_model, d_model)
            self.W_v     = nn.Linear(d_model, d_model)
            self.W_o     = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self._scale  = math.sqrt(self.d_k)

        def _split(self, x: Tensor) -> Tensor:
            B, S, _ = x.shape
            return x.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        def _merge(self, x: Tensor) -> Tensor:
            B, _, S, _ = x.shape
            return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

        def forward(
            self,
            query: Tensor,
            key:   Tensor,
            value: Tensor,
            mask:  Tensor | None = None,
        ) -> Tensor:
            Q = self._split(self.W_q(query))
            K = self._split(self.W_k(key))
            V = self._split(self.W_v(value))
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = self.dropout(F.softmax(scores, dim=-1))
            return self.W_o(self._merge(torch.matmul(w, V)))


    class FeedForward(nn.Module):
        """Position-wise two-layer FFN with ReLU."""

        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    # ------------------------------------------------------------------
    # Encoder block (Pre-LN)
    # ------------------------------------------------------------------

    class EncoderBlock(nn.Module):
        """
        Transformer encoder block with Pre-LN.

        Data flow:
          x → LayerNorm → MultiHeadSelfAttn → residual add
            → LayerNorm → FFN              → residual add
        """

        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
            self.ffn   = FeedForward(d_model, d_ff, dropout)

        def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
            normed = self.norm1(x)
            x = x + self.attn(normed, normed, normed, src_mask)
            x = x + self.ffn(self.norm2(x))
            return x


    # ------------------------------------------------------------------
    # Decoder block (Pre-LN)
    # ------------------------------------------------------------------

    class DecoderBlock(nn.Module):
        """
        Transformer decoder block with Pre-LN.

        Data flow:
          x → LayerNorm → CausalSelfAttn          → residual add
            → LayerNorm → CrossAttn(enc_out)       → residual add
            → LayerNorm → FFN                      → residual add
        """

        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.norm1      = nn.LayerNorm(d_model)
            self.norm2      = nn.LayerNorm(d_model)
            self.norm3      = nn.LayerNorm(d_model)
            self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ffn        = FeedForward(d_model, d_ff, dropout)

        def forward(
            self,
            x:        Tensor,
            enc_out:  Tensor,
            tgt_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
        ) -> Tensor:
            # 1. Causal self-attention over decoder prefix
            xn = self.norm1(x)
            x  = x + self.self_attn(xn, xn, xn, tgt_mask)
            # 2. Cross-attention: Q from decoder, K/V from encoder
            xn = self.norm2(x)
            x  = x + self.cross_attn(xn, enc_out, enc_out, src_mask)
            # 3. FFN
            x  = x + self.ffn(self.norm3(x))
            return x


    # ------------------------------------------------------------------
    # Encoder stack
    # ------------------------------------------------------------------

    class TransformerEncoder(nn.Module):
        """Stack of N encoder blocks with token embedding and PE."""

        def __init__(
            self,
            vocab_size: int,
            d_model:    int,
            n_heads:    int,
            n_layers:   int,
            d_ff:       int,
            dropout:    float = 0.1,
            max_len:    int   = 5_000,
            pad_idx:    int   = 0,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
            self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.layers    = nn.ModuleList([
                EncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm  = nn.LayerNorm(d_model)   # Pre-LN: final normalisation
            self.scale = math.sqrt(d_model)

        def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
            x = self.pos_enc(self.embedding(src) * self.scale)
            for layer in self.layers:
                x = layer(x, src_mask)
            return self.norm(x)


    # ------------------------------------------------------------------
    # Decoder stack
    # ------------------------------------------------------------------

    class TransformerDecoder(nn.Module):
        """Stack of N decoder blocks with token embedding, PE, and output projection."""

        def __init__(
            self,
            vocab_size: int,
            d_model:    int,
            n_heads:    int,
            n_layers:   int,
            d_ff:       int,
            dropout:    float = 0.1,
            max_len:    int   = 5_000,
            pad_idx:    int   = 0,
        ) -> None:
            super().__init__()
            self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
            self.pos_enc     = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.layers      = nn.ModuleList([
                DecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm        = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, vocab_size)
            self.scale       = math.sqrt(d_model)

        def forward(
            self,
            tgt:      Tensor,
            enc_out:  Tensor,
            tgt_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
        ) -> Tensor:
            x = self.pos_enc(self.embedding(tgt) * self.scale)
            for layer in self.layers:
                x = layer(x, enc_out, tgt_mask, src_mask)
            return self.output_proj(self.norm(x))   # (B, T, vocab_size)


    # ------------------------------------------------------------------
    # Full Transformer (encoder-decoder)
    # ------------------------------------------------------------------

    class Transformer(nn.Module):
        """
        Complete encoder-decoder Transformer for sequence-to-sequence tasks.

        Parameters
        ----------
        src_vocab_size    : Source vocabulary size.
        tgt_vocab_size    : Target vocabulary size.
        d_model           : Embedding / model dimension.
        n_heads           : Number of attention heads.
        n_encoder_layers  : Number of stacked encoder blocks.
        n_decoder_layers  : Number of stacked decoder blocks.
        d_ff              : FFN inner dimension (default: 4 × d_model).
        dropout           : Dropout probability.
        max_len           : Maximum sequence length.
        pad_idx           : Padding token index.
        """

        def __init__(
            self,
            src_vocab_size:   int,
            tgt_vocab_size:   int,
            d_model:          int   = 512,
            n_heads:          int   = 8,
            n_encoder_layers: int   = 6,
            n_decoder_layers: int   = 6,
            d_ff:             int   = 2_048,
            dropout:          float = 0.1,
            max_len:          int   = 5_000,
            pad_idx:          int   = 0,
        ) -> None:
            super().__init__()
            self.encoder = TransformerEncoder(
                src_vocab_size, d_model, n_heads, n_encoder_layers,
                d_ff, dropout, max_len, pad_idx,
            )
            self.decoder = TransformerDecoder(
                tgt_vocab_size, d_model, n_heads, n_decoder_layers,
                d_ff, dropout, max_len, pad_idx,
            )
            self._init_weights()

        def _init_weights(self) -> None:
            """Xavier uniform initialisation for weight matrices."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        @staticmethod
        def make_src_mask(src: Tensor, pad_idx: int = 0) -> Tensor:
            """
            Source padding mask.

            Returns
            -------
            (B, 1, 1, S_src) boolean — True where the token is NOT padding.
            """
            return (src != pad_idx).unsqueeze(1).unsqueeze(2)

        @staticmethod
        def make_tgt_mask(tgt: Tensor, pad_idx: int = 0) -> Tensor:
            """
            Combined causal + padding mask for the target sequence.

            Returns
            -------
            (B, 1, S_tgt, S_tgt) boolean — True where attention is PERMITTED.
            """
            B, T   = tgt.shape
            pad_m  = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)          # (B, 1, 1, T)
            causal = torch.tril(
                torch.ones(T, T, device=tgt.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)                                    # (1, 1, T, T)
            return pad_m & causal

        def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
            """
            Teacher-forcing forward pass.

            Parameters
            ----------
            src : (B, S_src) source token IDs.
            tgt : (B, S_tgt) target input token IDs (shifted right, starting with BOS).

            Returns
            -------
            logits : (B, S_tgt, tgt_vocab_size)
            """
            src_mask = self.make_src_mask(src)
            tgt_mask = self.make_tgt_mask(tgt)
            enc_out  = self.encoder(src, src_mask)
            return self.decoder(tgt, enc_out, tgt_mask, src_mask)


    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    if __name__ == "__main__":
        torch.manual_seed(0)

        model = Transformer(
            src_vocab_size=1_000,
            tgt_vocab_size=1_200,
            d_model=128, n_heads=4,
            n_encoder_layers=3, n_decoder_layers=3,
            d_ff=512, dropout=0.1,
        )

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")

        src = torch.randint(1, 1_000, (2, 20))   # batch=2, src_len=20
        tgt = torch.randint(1, 1_200, (2, 15))   # batch=2, tgt_len=15
        logits = model(src, tgt)
        print(f"Output logits: {logits.shape}")   # (2, 15, 1200)
    ```

---

## 2.7 Training: Tokenisation, Teacher Forcing, and Label Smoothing

### 2.7.1 Tokenisation

Raw text is converted to integer token IDs before training. Common strategies:

| Strategy | Vocabulary | Used by | Notes |
|---|---|---|---|
| Word-level | ~50K–200K | Early NMT | Large vocab, OOV issues |
| BPE | 8K–50K | GPT, RoBERTa | Frequency-based merges |
| WordPiece | 30K | BERT | Likelihood-based merges |
| Unigram | 8K–32K | T5, LLaMA | Probabilistic segmentation |
| Byte-level BPE | 50K | GPT-2, GPT-4 | No OOV — all bytes covered |

### 2.7.2 Teacher Forcing

During training, the decoder receives **ground-truth tokens** as context at each step rather than its own previous predictions. The target input is the reference sequence shifted right by one position (prepended with `[BOS]`):

```
Source:     "Hello world"
Target in:  [BOS] "Bonjour" "monde"        ← fed to decoder
Target out:       "Bonjour" "monde" [EOS]  ← loss computed against these
```

Teacher forcing gives a strong, unambiguous training signal but creates **exposure bias**: at inference the decoder must condition on its own (potentially wrong) outputs, a distribution not seen during training. Scheduled sampling addresses this but adds complexity.

### 2.7.3 Label Smoothing Cross-Entropy

Standard one-hot cross-entropy trains the model to assign probability arbitrarily close to 1.0 for the correct token, which can over-fit and produce poorly calibrated probabilities. **Label smoothing** (Szegedy et al., 2016) softens the target distribution:

$$y_{\text{smooth}} = (1 - \varepsilon)\, y_{\text{one-hot}} + \frac{\varepsilon}{K}$$

where $\varepsilon = 0.1$ (used in the original Transformer paper) and $K$ is the vocabulary size. This regularises the output distribution and consistently improves translation BLEU scores by 0.5–1.0 points.

=== "Python"

    ```python
    from __future__ import annotations

    import torch
    import torch.nn as nn
    from torch import Tensor


    class LabelSmoothingCrossEntropy(nn.Module):
        """
        Cross-entropy with label smoothing.

        Parameters
        ----------
        vocab_size : Number of output classes.
        pad_idx    : Index of the padding token (excluded from loss).
        smoothing  : Label smoothing factor ε. Default 0.1.
        """

        def __init__(
            self,
            vocab_size: int,
            pad_idx:    int   = 0,
            smoothing:  float = 0.1,
        ) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.pad_idx    = pad_idx
            self.smoothing  = smoothing
            self.confidence = 1.0 - smoothing

        def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
            """
            Parameters
            ----------
            logits  : (N, vocab_size) — raw model output, typically flattened (B*T, V).
            targets : (N,) — gold token IDs.

            Returns
            -------
            Scalar mean loss, ignoring padding positions.
            """
            log_probs = torch.log_softmax(logits, dim=-1)

            # Build smooth target distribution
            smooth_dist = torch.full_like(
                log_probs, self.smoothing / (self.vocab_size - 2)
            )
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0   # padding never contributes

            # Zero out padding positions in the loss
            pad_mask         = targets == self.pad_idx
            loss             = (-smooth_dist * log_probs).sum(dim=-1)
            loss[pad_mask]   = 0.0
            n_active         = (~pad_mask).sum().clamp(min=1)
            return loss.sum() / n_active.float()
    ```

---

## 2.8 Hyperparameter Reference

| Configuration | $d_{\text{model}}$ | $n_{\text{heads}}$ | $n_{\text{layers}}$ | $d_{ff}$ | Dropout | Approx. params |
|---|---|---|---|---|---|---|
| Small | 256 | 4 | 4 | 1,024 | 0.1 | ~10M |
| Base | 512 | 8 | 6 | 2,048 | 0.1 | ~65M |
| Large | 1,024 | 16 | 12 | 4,096 | 0.1 | ~300M |
| BERT-Base | 768 | 12 | 12 | 3,072 | 0.1 | ~110M |
| GPT-2 Medium | 1,024 | 16 | 24 | 4,096 | 0.1 | ~345M |
| GPT-3 175B | 12,288 | 96 | 96 | 49,152 | 0.0 | ~175B |

**Design rules of thumb**:

- $d_{ff} = 4 \times d_{\text{model}}$ is the standard ratio; SwiGLU uses $\approx 2.67 \times d_{\text{model}}$ for three matrices.
- $d_k = d_{\text{model}} / n_{\text{heads}}$ should be at least 32 for numerically stable softmax.
- At large scale (>1B parameters), dropout is reduced to 0.0 — dataset size provides the regularisation.
- Learning rate warmup (typically 4,000–10,000 steps with a cosine or inverse-square-root schedule) is important especially for Post-LN models.

---

## 2.9 Exercises

**Exercise 2.1 — Positional Encoding Visualisation**  
Implement `SinusoidalPositionalEncoding` and visualise the encoding matrix as a heatmap (positions 0–99 on y-axis, dimensions 0–63 on x-axis) for $d_{\text{model}} = 64$. Explain the pattern: why do slow-varying (low-frequency) sinusoids appear in higher dimension indices, and what property allows the model to extract relative positions from absolute encodings?

**Exercise 2.2 — Pre-LN vs Post-LN Gradient Norms**  
Implement both `EncoderBlock` variants (Pre-LN and Post-LN). For each, build a 12-layer encoder stack, run a random forward pass, backpropagate a scalar loss, and use `p.grad.norm()` to plot the gradient norm at each layer's input. Verify empirically that Pre-LN produces more uniform gradient norms across depth and explain why this matters for stable training.

**Exercise 2.3 — Label Smoothing Ablation**  
Train two `Transformer` instances on a toy word-reversal task (reverse sequences of random integers in [1, 100] of length 10–20): one with hard cross-entropy ($\varepsilon = 0$) and one with label smoothing ($\varepsilon = 0.1$). Plot training and validation accuracy. Compare the predicted probability distributions (entropy of the output softmax) on 10 held-out examples — does label smoothing produce more diffuse outputs?

**Exercise 2.4 — Attention Map Extraction**  
Modify `MultiHeadAttention.forward` to optionally return attention weight tensors. Train the `Transformer` (small configuration) on any seq2seq dataset. Extract and visualise the **cross-attention** weights for 3 held-out examples across all decoder layers and heads. Identify which heads and layers show the clearest input-output alignment.

**Exercise 2.5 — Full Transformer from Scratch**  
Implement the complete Transformer encoder-decoder from scratch — without referencing this chapter's code — derive every component from the Vaswani et al. paper. Train it on the `Multi30k` dataset (German → English translation via `torchtext`). Target: BLEU ≥ 20 on the validation set. Report: model configuration, training curve (train/val cross-entropy vs. epoch), and 10 example translations generated with greedy decoding and beam search ($k = 4$). Compare beam vs. greedy BLEU scores.

---

## Summary

The Transformer architecture assembles attention, positional encoding, and residual blocks into a complete, parallelisable, and scalable sequence model.

**Positional encoding** solves the permutation-equivariance problem. Sinusoidal PE is fixed and generalises to unseen lengths. Learned PE is more flexible. RoPE encodes positions as rotations of Q/K vectors, enabling principled long-context extension.

**Encoder blocks** apply all-to-all self-attention followed by a position-wise FFN, each wrapped in Add&Norm. **Decoder blocks** add causal self-attention (masked) and cross-attention before the FFN.

**Pre-LN** (LayerNorm before each sublayer) yields stable gradient flow across deep stacks and is the de facto standard in modern large models.

**Teacher forcing** with **label-smoothed cross-entropy** forms the standard training recipe. Label smoothing regularises the output distribution and consistently improves translation quality.

| Component | Key formula |
|---|---|
| Sinusoidal PE (even dim) | $\sin\!\left(\text{pos}/10000^{2i/d_{\text{model}}}\right)$ |
| Sinusoidal PE (odd dim) | $\cos\!\left(\text{pos}/10000^{2i/d_{\text{model}}}\right)$ |
| Add&Norm (Post-LN) | $\text{LayerNorm}(x + \text{Sublayer}(x))$ |
| Add&Norm (Pre-LN) | $x + \text{Sublayer}(\text{LayerNorm}(x))$ |
| FFN | $\max(0, xW_1 + b_1)W_2 + b_2$ |
| FFN parameters per layer | $2 d_{\text{model}} d_{ff}$ |
| MHA parameters per layer | $4 d_{\text{model}}^2$ |
| Total params (base, $N$=6) | $N(4d^2 + 2dd_{ff}) \approx 65\text{M}$ for base |

!!! tip "Next steps"
    Chapter 3 moves from building Transformers to using pre-trained ones: BERT's masked language modelling pre-training, GPT's autoregressive pre-training, the Kaplan et al. neural scaling laws, and practical fine-tuning including LoRA and prefix tuning for efficient adaptation to downstream tasks.
