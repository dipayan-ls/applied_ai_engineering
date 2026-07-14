# Glossary

A comprehensive reference of terms used throughout this textbook. Entries are precise — use this as a technical dictionary, not a simplified overview.

---

## A

**Activation Function**
:   A non-linear function applied element-wise to the output of a neural network layer. Common examples: ReLU, GeLU, Sigmoid, Tanh. Without non-linearity, a deep network collapses to a single linear transformation.

**Attention (Mechanism)**
:   A learned weighted combination of a set of value vectors, where weights are determined by the compatibility between query and key vectors. Formally: \(\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V\).

**Autoregressive Model**
:   A model that generates outputs one token at a time, conditioning each new token on all previously generated tokens. GPT-family models are autoregressive.

---

## B

**Backpropagation**
:   The algorithm for computing gradients of the loss with respect to all parameters in a neural network, using the chain rule of calculus. Operates in a backward pass after the forward pass.

**Batch Normalisation**
:   A technique that normalises layer inputs to zero mean and unit variance within a mini-batch, then applies learnable scale and shift parameters. Accelerates training and provides mild regularisation.

**Bayes' Theorem**
:   \(P(A|B) = P(B|A)P(A) / P(B)\). In ML contexts, relates prior beliefs about parameters to posterior beliefs after observing data.

**Beam Search**
:   A decoding algorithm that maintains *k* candidate sequences (the beam) at each step, expanding each with all possible next tokens and retaining only the top *k* by cumulative log probability.

**Bias-Variance Tradeoff**
:   The decomposition of expected prediction error into bias\(^2\) (systematic error from incorrect model assumptions), variance (sensitivity to training set fluctuations), and irreducible noise.

---

## C

**Causal Masking**
:   In decoder-only Transformers, a triangular mask applied to the attention matrix ensuring position *i* can only attend to positions ≤ *i*. Enforces the autoregressive property.

**Chain Rule**
:   The calculus identity \(\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)\). The mathematical foundation of backpropagation.

**Context Window**
:   The maximum number of tokens an LLM can process in a single forward pass. Determined by the positional encoding scheme and memory constraints.

**Cross-Entropy**
:   A loss function for classification: \(H(P, Q) = -\sum_x P(x) \log Q(x)\). Measures the average bits needed to encode samples from *P* using a code optimised for *Q*.

**Cross-Validation (k-fold)**
:   A model evaluation technique that partitions the dataset into *k* folds, trains on *k-1* folds, evaluates on the remaining fold, and averages results across all *k* rotations.

---

## D

**Data Drift**
:   A change in the statistical distribution of input features over time, which can degrade a deployed model's performance without any change to the model itself.

**Direct Preference Optimisation (DPO)**
:   An RLHF alternative that fine-tunes a language model directly on preference pairs (preferred vs rejected response) without requiring an explicit reward model or PPO training loop.

**Dropout**
:   A regularisation technique that randomly zeroes activations during training (with probability *p*) and scales remaining activations by *1/(1-p)* (inverted dropout). Disabled at inference.

---

## E

**Embedding**
:   A dense, low-dimensional vector representation of a discrete object (token, sentence, entity). Embeddings encode semantic similarity: similar items have similar vectors.

**Entropy (Shannon)**
:   \(H(X) = -\sum_x P(x) \log_2 P(x)\). Measures the average information content (bits) of outcomes from a probability distribution.

---

## F

**Feature Engineering**
:   The process of transforming raw data into informative numeric representations suitable for ML models. Includes normalisation, encoding, polynomial features, and domain-specific transformations.

**Few-Shot Learning**
:   Learning from a small number of labelled examples. In LLM contexts, refers to providing examples in the prompt to guide model behaviour without weight updates.

**Fine-Tuning**
:   Continuing training of a pre-trained model on task-specific data. Typically uses a smaller learning rate than pre-training to avoid catastrophic forgetting.

**Flashattention**
:   An IO-aware exact attention algorithm that computes attention in tiles, avoiding materialising the full \(N \times N\) attention matrix, achieving \(O(N)\) memory and significant GPU speedup.

---

## G

**Generative Adversarial Network (GAN)**
:   A framework with a generator \(G\) and discriminator \(D\) trained in opposition: \(G\) generates fake samples to fool \(D\); \(D\) learns to distinguish real from fake. Introduced by Goodfellow et al. (2014).

**Gradient**
:   A vector \(\nabla_\theta L\) of partial derivatives of the loss with respect to each parameter. Points in the direction of steepest increase; gradient descent moves opposite to it.

**Gradient Descent**
:   An iterative optimisation algorithm: \(\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)\). Variants include batch, mini-batch (SGD), and stochastic gradient descent.

---

## H

**Hallucination**
:   LLM-generated output that is factually incorrect, ungrounded, or fabricated but stated with apparent confidence. A key failure mode in production LLM systems.

**HNSW (Hierarchical Navigable Small Worlds)**
:   An approximate nearest neighbour (ANN) index structure based on a hierarchical graph. Achieves sub-linear query time with high recall; the dominant algorithm in production vector databases.

---

## K

**KL Divergence**
:   \(D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}\). Measures how much distribution *P* differs from *Q*. Not symmetric. Used in variational autoencoders and RLHF KL penalties.

**KV Cache**
:   In Transformer inference, previously computed key and value matrices are cached and reused for each new token generation step. Reduces computation from \(O(n^2)\) per new token to \(O(n)\), at the cost of \(O(n \cdot d_{model})\) memory per sequence.

---

## L

**Layer Normalisation**
:   Normalises activations across the feature dimension (not batch dimension) of a single sample: \(\hat{x}_i = (x_i - \mu) / (\sigma + \epsilon)\). More stable than batch norm for variable-length sequences; standard in Transformers.

**LoRA (Low-Rank Adaptation)**
:   Parameter-efficient fine-tuning method that freezes pre-trained weights and adds low-rank decomposition matrices \(\Delta W = BA\) (where rank \(r \ll d\)) to each targeted weight matrix.

---

## M

**Maximum Likelihood Estimation (MLE)**
:   \(\hat{\theta} = \arg\max_\theta \log P(\mathcal{D}|\theta)\). Finds parameters that maximise the probability of observing the training data. Pre-training of language models uses MLE on next-token prediction.

**Multi-Head Attention**
:   Runs *h* parallel attention heads, each with its own projection matrices, concatenates their outputs, and applies a final projection. Allows the model to attend to different representation subspaces simultaneously.

---

## N

**Nucleus Sampling (Top-p)**
:   A decoding strategy that samples from the smallest set of tokens whose cumulative probability exceeds *p*. Balances diversity and coherence better than top-k for variable vocabulary richness.

---

## O

**Overfitting**
:   When a model learns the training data too closely, including noise and irrelevant patterns, and fails to generalise to unseen data. Detected by a gap between training and validation loss.

---

## P

**Perplexity**
:   \(\text{PPL} = e^{H(P, Q)}\). Measures how well a language model predicts a test set. Lower is better. A perplexity of *k* means the model is as uncertain as if choosing uniformly among *k* options at each step.

**Positional Encoding**
:   A mechanism to inject sequence order information into a Transformer (which is inherently permutation-invariant). Implementations: sinusoidal (fixed), learned, RoPE (Rotary Position Embedding), ALiBi.

---

## R

**RAG (Retrieval-Augmented Generation)**
:   An architecture that augments an LLM's generation by first retrieving relevant documents from an external knowledge store, then conditioning the LLM's response on retrieved context.

**Regularisation**
:   Techniques that reduce overfitting by adding a penalty for model complexity. L2 (Ridge): \(\lambda ||w||^2\). L1 (Lasso): \(\lambda ||w||_1\). Dropout and early stopping are also regularisation techniques.

**RLHF (Reinforcement Learning from Human Feedback)**
:   A training pipeline: (1) supervised fine-tuning on demonstrations; (2) reward model trained on human preference pairs; (3) LLM optimised with PPO to maximise reward subject to a KL penalty.

**RoPE (Rotary Position Embedding)**
:   A positional encoding method that encodes absolute position as a rotation in embedding space and naturally enables relative positional attention. Used in LLaMA, Mistral, and Gemma.

---

## S

**Scaling Laws**
:   Empirical relationships between model performance and compute, dataset size, and parameter count. Kaplan et al. (2020) showed power-law scaling; Hoffmann et al. (2022, Chinchilla) showed compute-optimal data/parameter ratios.

**Singular Value Decomposition (SVD)**
:   \(A = U \Sigma V^T\). Decomposes a matrix into orthogonal bases and singular values. Foundational for PCA, low-rank approximations, and the analysis of weight matrices in LLMs.

**Softmax**
:   \(\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}\). Converts a vector of real numbers into a probability distribution. Used in attention and output classification layers.

---

## T

**Temperature**
:   A parameter that scales logits before softmax: \(\text{softmax}(z/T)\). High temperature → more uniform distribution → more diverse outputs. Low temperature → sharper distribution → more deterministic outputs. *T=0* is equivalent to greedy decoding.

**Tokenisation**
:   The process of splitting text into a sequence of tokens (sub-words, words, or characters) and mapping them to integer IDs. BPE, WordPiece, and SentencePiece are common algorithms.

**Transformer**
:   A neural architecture relying entirely on attention mechanisms (no recurrence or convolution). Introduced in "Attention Is All You Need" (Vaswani et al., 2017). The dominant architecture for LLMs, vision, and multi-modal models.

---

## V

**Vector Database**
:   A storage system optimised for efficient similarity search over high-dimensional embedding vectors. Examples: Qdrant, Pinecone, Weaviate, Chroma, pgvector.

**Vanishing Gradient**
:   When gradients become exponentially small as they are backpropagated through many layers, making early layers train very slowly or not at all. Addressed by residual connections, LSTM gates, and proper weight initialisation.
