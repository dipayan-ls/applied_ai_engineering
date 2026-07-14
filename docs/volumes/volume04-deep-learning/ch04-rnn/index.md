---
title: "Ch 4 — Recurrent Networks & LSTMs"
---

# Ch 4 — Recurrent Networks & LSTMs

<div class="chapter-meta" markdown>
| | |
|---|---|
| **Difficulty** | Advanced |
| **Reading time** | 90 min |
| **Prerequisites** | Ch 1 (Neural Networks), Ch 2 (Training Deep Networks), Ch 3 (Convolutional Networks) |
</div>

---

## Learning Objectives

By the end of this chapter you will be able to:

1. **Derive the RNN cell equations** from first principles and explain why the same weight matrix is shared across all time steps.
2. **Explain vanishing and exploding gradients** in RNNs using the chain rule, quantify the exponential decay/growth over long sequences, and describe gradient clipping as a practical remedy.
3. **Implement an LSTM** from scratch in PyTorch, correctly applying all four gate equations and the cell-state update, and justify each gate's role as a learned memory controller.
4. **Compare RNNs, LSTMs, and GRUs** across the dimensions of parameter count, training speed, and empirical performance on short vs. long sequences, making a principled choice for a given task.
5. **Build a complete sequence classification pipeline** in PyTorch using `nn.LSTM`, packed sequences for variable-length inputs, and a classification head — trained and evaluated on a sentiment analysis task.

---

## 4.1 The Recurrent Neural Network Cell

### 4.1.1 Motivation: Sequence Data

Standard feedforward networks assume inputs are independent — each example is processed in isolation. Many real-world problems violate this assumption fundamentally:

- In **natural language**, the word "bank" means different things in "river bank" vs. "investment bank" — context is everything.
- In **time series**, today's stock price is correlated with yesterday's.
- In **audio**, phoneme boundaries only make sense relative to surrounding frames.

A **Recurrent Neural Network (RNN)** addresses this by maintaining a **hidden state** $h_t$ that acts as a running summary of the sequence seen so far. The same set of weights is applied at every time step, giving the network an inductive bias for temporal patterns.

### 4.1.2 The RNN Cell Equations

At each time step $t$, the RNN takes the current input $x_t$ and the previous hidden state $h_{t-1}$ and produces a new hidden state $h_t$ and output $y_t$:

$$h_t = \tanh\!\left(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\right)$$

$$y_t = W_{hy}\, h_t + b_y$$

**Dimensions** (for a model with input size $d_x$, hidden size $d_h$, output size $d_y$):

| Parameter | Shape | Purpose |
|---|---|---|
| $W_{xh}$ | $d_h \times d_x$ | Projects input into hidden space |
| $W_{hh}$ | $d_h \times d_h$ | Recurrent transition weights |
| $b_h$ | $d_h$ | Hidden bias |
| $W_{hy}$ | $d_y \times d_h$ | Output projection |
| $b_y$ | $d_y$ | Output bias |

The tanh activation squashes values into $[-1, 1]$, preventing unbounded growth of activations. Weight sharing across time steps means the number of parameters does not grow with sequence length.

!!! note "Parameter counting"
    An RNN with $d_x = 100$, $d_h = 256$, $d_y = 10$ has only $256 \times 100 + 256 \times 256 + 256 + 10 \times 256 + 10 \approx 95K$ parameters regardless of whether the sequence is 10 or 10,000 tokens long.

---

## 4.2 Backpropagation Through Time (BPTT)

### 4.2.1 Unrolling the Computation Graph

To train an RNN with gradient descent, we **unroll** the network across time steps, creating a computational graph that looks like a deep feedforward network with shared weights. For a sequence of length $T$, the loss is:

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(y_t, \hat{y}_t)$$

The gradient of the loss with respect to the hidden-to-hidden weights $W_{hh}$ requires backpropagating through all $T$ time steps:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hh}}$$

Each term $\frac{\partial \mathcal{L}_t}{\partial W_{hh}}$ involves a chain of Jacobians through all hidden states from step 1 to step $t$:

$$\frac{\partial \mathcal{L}_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{t} W_{hh}^T \cdot \text{diag}\!\left(\tanh'\!(z_k)\right)$$

where $z_k = W_{hh} h_{k-1} + W_{xh} x_k + b_h$ and $\tanh'(z) = 1 - \tanh^2(z)$.

### 4.2.2 The Vanishing Gradient Problem

This product of $T-1$ Jacobian matrices is the source of the famous **vanishing gradient problem**. Consider the spectral (largest singular value) norm of each factor:

$$\left\|\frac{\partial h_k}{\partial h_{k-1}}\right\|_2 = \|W_{hh}^T \cdot \text{diag}(\tanh'(z_k))\|_2 \leq \|W_{hh}\|_2 \cdot \max_j |\tanh'(z_{k,j})|$$

Since $\tanh'(z) \leq 1$ (with equality only at $z = 0$), if $\|W_{hh}\|_2 < 1$ then:

$$\left\|\prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}}\right\|_2 \leq \gamma^{t-1}, \quad \gamma < 1$$

The gradient **decays exponentially** with sequence length. For $T = 100$ and $\gamma = 0.9$, the gradient at step 1 is $\approx 0.9^{99} \approx 2.7 \times 10^{-5}$ of the gradient at step $T$.

**Practical consequence**: vanilla RNNs cannot learn dependencies spanning more than roughly 10–20 time steps.

### 4.2.3 The Exploding Gradient Problem

If $\|W_{hh}\|_2 > 1$, the product grows exponentially, causing numerical overflow and erratic training. The standard remedy is **gradient clipping** — rescaling the gradient norm if it exceeds a threshold $\theta$:

$$g \leftarrow g \cdot \frac{\theta}{\|g\|_2} \quad \text{if } \|g\|_2 > \theta$$

Typical values: $\theta \in [1, 5]$.

!!! warning "Vanishing vs. exploding"
    Exploding gradients are *detectable* — training loss diverges or becomes NaN. Vanishing gradients are *silent* — training converges, but the model quietly ignores long-range dependencies. Always plot gradient norms during training.

---

## 4.3 Long Short-Term Memory (LSTM)

### 4.3.1 The Problem with Vanilla RNNs

The single hidden state $h_t$ must simultaneously serve as:
1. The memory of what happened earlier in the sequence.
2. The input to the output projection for the current step.

These conflicting roles, combined with the vanishing gradient problem, mean that vanilla RNNs effectively have short-term memory. Long-range dependencies are squeezed out by recent inputs.

### 4.3.2 The Cell State: A Memory Highway

Hochreiter & Schmidhuber (1997) introduced the LSTM to solve this. The key innovation is the **cell state** $C_t$ — a separate, slowly-changing memory that flows through the sequence with only minor, gated modifications. Gradients can flow through the cell state **without** passing through the tanh nonlinearity at every step, dramatically reducing vanishing gradient issues.

### 4.3.3 The Four Gate Equations

The LSTM uses four gating mechanisms, each a learned sigmoid-activated projection of $[h_{t-1}, x_t]$ (concatenation):

**Forget gate** — decides what to erase from the cell state:

$$f_t = \sigma\!\left(W_f \cdot [h_{t-1},\, x_t] + b_f\right)$$

Values near 0 erase; values near 1 retain. This gate is what allows the LSTM to "forget" an author's gender pronoun reference when a new subject is introduced.

**Input gate** — decides which new information to write to memory:

$$i_t = \sigma\!\left(W_i \cdot [h_{t-1},\, x_t] + b_i\right)$$

**Candidate cell values** — what new content to potentially add:

$$\tilde{C}_t = \tanh\!\left(W_C \cdot [h_{t-1},\, x_t] + b_C\right)$$

**Cell state update** — combine forget (erase old) and input (write new):

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

The Hadamard product $\odot$ applies gating element-wise. This is the memory highway: if $f_t \approx 1$ and $i_t \approx 0$, the cell state is almost unchanged and gradients flow freely.

**Output gate** — decides what part of the cell to expose as the hidden state:

$$o_t = \sigma\!\left(W_o \cdot [h_{t-1},\, x_t] + b_o\right)$$

$$h_t = o_t \odot \tanh(C_t)$$

The cell state is squashed through tanh to bound it to $[-1, 1]$ before being gated by $o_t$.

!!! note "Why four gates?"
    Each gate is a *soft switch* learned from data. The network discovers which inputs are worth remembering, how long to retain them, and which aspects to use for output — without these decisions being hard-coded by the programmer.

**Total LSTM parameter count** for input size $d_x$, hidden size $d_h$:

$$4 \times (d_h \cdot (d_h + d_x) + d_h) = 4 d_h (d_h + d_x + 1)$$

---

## 4.4 Gated Recurrent Units (GRU)

Cho et al. (2014) introduced the **GRU** as a simplified variant of the LSTM that uses only two gates and merges the cell state and hidden state into a single vector.

**Reset gate** — controls how much of the previous hidden state to mix with the new input:

$$r_t = \sigma\!\left(W_r \cdot [h_{t-1},\, x_t] + b_r\right)$$

**Update gate** — controls the interpolation between old and new hidden state:

$$z_t = \sigma\!\left(W_z \cdot [h_{t-1},\, x_t] + b_z\right)$$

**Candidate hidden state**:

$$\tilde{h}_t = \tanh\!\left(W \cdot [r_t \odot h_{t-1},\, x_t] + b\right)$$

**Final hidden state** — a soft interpolation:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

When $z_t \approx 0$, the hidden state is unchanged (like the LSTM forget gate). When $z_t \approx 1$, the new candidate replaces it entirely.

### 4.4.1 LSTM vs. GRU Comparison

| Property | LSTM | GRU |
|---|---|---|
| Gates | 4 (f, i, o, candidate) | 2 (reset, update) |
| Separate cell state | Yes ($C_t$) | No (merged into $h_t$) |
| Parameters | $4 d_h (d_h + d_x + 1)$ | $3 d_h (d_h + d_x + 1)$ |
| Training speed | Slower | ~25 % faster |
| Long sequences | Slightly better | Competitive |
| Typical use case | Sequence generation, language models | Short-medium sequences, fast baselines |

For most tasks with sequences under 500 steps, the empirical performance difference is negligible. GRU is a pragmatic choice when compute is constrained.

---

## 4.5 Bidirectional RNNs

A **unidirectional** RNN at step $t$ can only see the past: $x_1, \ldots, x_t$. For many tasks (POS tagging, NER, sentiment), the future context is equally important.

A **Bidirectional RNN** runs two independent RNNs over the same sequence — one forward, one backward — and concatenates their hidden states:

$$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})$$

$$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

$$h_t = \left[\overrightarrow{h}_t;\, \overleftarrow{h}_t\right]$$

The concatenated representation at step $t$ has access to the full sequence context, doubling the hidden size. Bidirectional LSTMs (BiLSTMs) were the dominant architecture for NLP tasks before Transformers.

!!! warning "Causality constraint"
    Bidirectional RNNs are **only applicable to offline/batch tasks** where the full sequence is available before prediction. For online tasks (speech streaming, language generation), only the forward direction can be used.

---

## 4.6 Sequence-to-Sequence with Bahdanau Attention

### 4.6.1 The Fixed-Size Bottleneck

The basic seq2seq model (Sutskever et al., 2014) encodes the full source sequence into a single context vector $c = h_T$ (the final encoder hidden state), then decodes from it. For long sequences this is catastrophic — a 500-word sentence must be compressed into a single vector, necessarily losing information.

### 4.6.2 Bahdanau Additive Attention

Bahdanau et al. (2015) introduced a mechanism allowing the decoder to **attend differently to each encoder step** at each decoding step.

For decoder step $t$ and encoder step $j$, the alignment score (how relevant encoder position $j$ is for decoder step $t$):

$$e_{tj} = v_a^T \tanh\!\left(W_a\, s_{t-1} + U_a\, \bar{h}_j\right)$$

where $s_{t-1}$ is the previous decoder hidden state and $\bar{h}_j$ is the encoder hidden state at position $j$.

The alignment weights (normalised via softmax):

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_x} \exp(e_{tk})}$$

The context vector is a weighted sum of all encoder hidden states:

$$c_t = \sum_{j=1}^{T_x} \alpha_{tj}\, \bar{h}_j$$

The decoder at step $t$ takes $(c_t, s_{t-1}, y_{t-1})$ as input to produce the next state $s_t$ and prediction $\hat{y}_t$.

The $\alpha_{tj}$ matrix is the **attention map** — it can be visualised as a soft alignment between source and target tokens, revealing what the model "looks at" when generating each output word.

---

## 4.7 Why Transformers Replaced RNNs

Despite LSTMs' success, they have two fundamental architectural limitations that Transformers (Vaswani et al., 2017) solved directly:

### 4.7.1 Sequential Computation — O(n) vs O(1)

RNNs are **inherently sequential**: $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, etc. This prevents parallelisation during training. For a sequence of length $n$, the encoder requires $O(n)$ sequential steps regardless of hardware. Transformers process all positions **simultaneously** via self-attention — $O(1)$ sequential steps.

### 4.7.2 Path Length for Long-Range Dependencies — O(n) vs O(1)

The number of computational steps between any two positions in the sequence determines how well the model can learn their dependency. In an RNN, information from position 1 must travel through $n-1$ sequential steps to reach position $n$ — an $O(n)$ path. In a Transformer, self-attention connects any two positions directly — an $O(1)$ path, independent of sequence length.

### 4.7.3 Summary Comparison

| Property | Vanilla RNN | LSTM / GRU | Transformer |
|---|---|---|---|
| Sequential computation | $O(n)$ | $O(n)$ | $O(1)$ |
| Max path length | $O(n)$ | $O(n)$ | $O(1)$ |
| Memory per step | $O(1)$ | $O(1)$ | $O(n^2)$ |
| Training parallelism | None | None | Full |
| Long-range learning | Poor | Moderate | Excellent |
| Parameter sharing | Across steps | Across steps | Across layers |

The $O(n^2)$ memory cost of the Transformer's attention matrix is the price paid for $O(1)$ path length — a trade-off that is worth it for most tasks with sequences up to ~100K tokens, and is being addressed by approximate attention methods (Flash Attention, Longformer, etc.).

---

## 4.8 PyTorch LSTM for Sentiment Classification

Below is a complete, typed, end-to-end PyTorch implementation of a bidirectional LSTM classifier for sentiment analysis using packed sequences to handle variable-length inputs efficiently.

=== "Python"

    ```python
    from __future__ import annotations

    from dataclasses import dataclass

    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset


    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    @dataclass
    class SentimentSample:
        """A single tokenised sentiment example."""
        token_ids: list[int]
        label: int       # 0 = negative, 1 = positive


    class SentimentDataset(Dataset[tuple[Tensor, int, int]]):
        """
        Dataset returning (padded_sequence, length, label) tuples.

        Parameters
        ----------
        samples    : List of SentimentSample objects.
        max_length : Maximum sequence length (longer sequences are truncated).
        """

        def __init__(
            self,
            samples: list[SentimentSample],
            max_length: int = 256,
        ) -> None:
            self.samples    = samples
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> tuple[Tensor, int, int]:
            sample = self.samples[idx]
            ids    = sample.token_ids[: self.max_length]
            length = len(ids)
            padded = torch.zeros(self.max_length, dtype=torch.long)
            padded[:length] = torch.tensor(ids, dtype=torch.long)
            return padded, length, sample.label


    def collate_fn(
        batch: list[tuple[Tensor, int, int]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sort batch by decreasing length for pack_padded_sequence."""
        seqs, lengths, labels = zip(*batch)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        # Sort by descending length
        sorted_idx = torch.argsort(lengths_t, descending=True)
        seqs_t    = torch.stack(seqs)[sorted_idx]
        lengths_t = lengths_t[sorted_idx]
        labels_t  = torch.tensor(labels, dtype=torch.long)[sorted_idx]
        return seqs_t, lengths_t, labels_t


    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------

    class BiLSTMClassifier(nn.Module):
        """
        Bidirectional LSTM sentiment classifier.

        Parameters
        ----------
        vocab_size    : Size of the vocabulary.
        embed_dim     : Embedding dimension.
        hidden_dim    : LSTM hidden state size (per direction).
        num_layers    : Number of stacked LSTM layers.
        num_classes   : Number of output classes.
        dropout       : Dropout probability (applied between LSTM layers).
        padding_idx   : Index of the padding token.
        """

        def __init__(
            self,
            vocab_size:   int,
            embed_dim:    int   = 128,
            hidden_dim:   int   = 256,
            num_layers:   int   = 2,
            num_classes:  int   = 2,
            dropout:      float = 0.3,
            padding_idx:  int   = 0,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            # Bidirectional → hidden_dim * 2 features
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        def forward(
            self,
            token_ids: Tensor,     # (batch, max_len)
            lengths:   Tensor,     # (batch,)
        ) -> Tensor:               # (batch, num_classes) — raw logits
            embedded = self.dropout(self.embedding(token_ids))  # (B, L, E)

            # Pack to skip computation on padding tokens
            packed: PackedSequence = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )
            packed_out, (h_n, _) = self.lstm(packed)

            # h_n shape: (num_layers * num_directions, batch, hidden_dim)
            # Take the last layer's forward and backward final states
            h_fwd = h_n[-2]   # forward direction, last layer
            h_bwd = h_n[-1]   # backward direction, last layer
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)   # (B, hidden_dim * 2)

            return self.classifier(self.dropout(h_cat))


    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    def train_epoch(
        model:      BiLSTMClassifier,
        loader:     DataLoader,
        optimiser:  torch.optim.Optimizer,
        criterion:  nn.CrossEntropyLoss,
        device:     torch.device,
    ) -> float:
        """Run one training epoch and return mean loss."""
        model.train()
        total_loss = 0.0
        for seqs, lengths, labels in loader:
            seqs    = seqs.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)

            optimiser.zero_grad()
            logits = model(seqs, lengths)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    @torch.no_grad()
    def evaluate(
        model:     BiLSTMClassifier,
        loader:    DataLoader,
        criterion: nn.CrossEntropyLoss,
        device:    torch.device,
    ) -> tuple[float, float]:
        """Return (mean_loss, accuracy)."""
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for seqs, lengths, labels in loader:
            seqs    = seqs.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)
            logits  = model(seqs, lengths)
            total_loss += criterion(logits, labels).item()
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return total_loss / len(loader), correct / total


    # ---------------------------------------------------------------------------
    # Quick sanity check (runs without real data)
    # ---------------------------------------------------------------------------
    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Synthetic data: vocab size 5000, 200 samples, variable lengths
        rng = torch.Generator().manual_seed(42)
        fake_samples = [
            SentimentSample(
                token_ids=torch.randint(1, 5000, (torch.randint(20, 200, (1,), generator=rng).item(),), generator=rng).tolist(),
                label=torch.randint(0, 2, (1,), generator=rng).item(),
            )
            for _ in range(200)
        ]

        dataset = SentimentDataset(fake_samples, max_length=200)
        loader  = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

        model     = BiLSTMClassifier(vocab_size=5000).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(3):
            loss = train_epoch(model, loader, optimiser, criterion, device)
            val_loss, acc = evaluate(model, loader, criterion, device)
            print(f"Epoch {epoch+1} | train loss: {loss:.4f} | val acc: {acc:.4f}")
    ```

---

## 4.9 Exercises

**Exercise 4.1 — BPTT by Hand**  
Consider a 3-step RNN with $d_h = 1$, $W_{hh} = 0.5$, $W_{xh} = 1.0$, $b_h = 0$, and initial hidden state $h_0 = 0$. Given inputs $x_1 = 1, x_2 = 2, x_3 = 3$, compute $h_1, h_2, h_3$ by hand (using tanh). Then compute $\partial h_3 / \partial h_1$ and explain whether this represents vanishing or stable gradient flow.

**Exercise 4.2 — LSTM Gate Ablation**  
Implement a simplified LSTM that removes the forget gate (fix $f_t = 1$). Train both the full LSTM and the ablated version on a synthetic counting task (e.g., count how many times token 5 appeared in the last 50 steps). Plot the learning curves. What does this reveal about the forget gate's role?

**Exercise 4.3 — GRU vs LSTM Benchmark**  
Using the `torchtext` IMDB dataset, train an LSTM and a GRU with the same hidden dimension ($d_h = 128$), same number of parameters (adjust depths accordingly), and same training budget (10 epochs). Report: test accuracy, training time per epoch, and gradient norm statistics. Discuss under what conditions you would prefer one over the other.

**Exercise 4.4 — Packed Sequences**  
Modify the `BiLSTMClassifier` to support `enforce_sorted=False` in `pack_padded_sequence`. Measure the overhead of the internal sort vs. pre-sorting the batch. Why does PyTorch originally require sorted sequences for efficient GPU kernels?

**Exercise 4.5 — Attention Visualisation**  
Extend the BiLSTM with Bahdanau additive attention (pooling over encoder steps instead of using only the final hidden state). Train on IMDB. Extract and visualise the $\alpha_{tj}$ attention weights for 5 example sentences. Do the words with highest attention correspond to semantically meaningful sentiment indicators?

---

## Summary

Recurrent networks process sequences by maintaining a hidden state that evolves over time. The vanilla RNN cell is elegant but plagued by vanishing gradients — exponential decay of gradient signal over long sequences prevents learning long-range dependencies.

The **LSTM** solves this with a dedicated cell state (the memory highway), gated by four learned controllers: forget, input, candidate, and output gates. The **GRU** achieves similar performance with fewer parameters using just reset and update gates.

**Bidirectional RNNs** double representational power for offline tasks by combining forward and backward passes. **Bahdanau attention** breaks the fixed-size bottleneck in seq2seq models, allowing the decoder to selectively attend to encoder positions at each decoding step.

Despite these advances, RNNs are fundamentally sequential ($O(n)$ computation) and have $O(n)$ path length between distant positions. **Transformers** address both limitations with self-attention, achieving $O(1)$ parallel computation and $O(1)$ path length at the cost of $O(n^2)$ memory — the trade-off that dominates modern sequence modelling.

!!! tip "Next steps"
    Volume 5 begins with the Attention Mechanism in Chapter 1, deriving scaled dot-product attention from the Bahdanau precursor and building toward the full Transformer architecture.
