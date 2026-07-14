# Quizzes

Each chapter has an associated quiz. Quizzes contain four question types:

1. **Multiple Choice** — Test conceptual understanding
2. **Short Answer** — Explain reasoning or derive a result
3. **Coding** — Write or debug Python/PyTorch code
4. **System Design** — Design a production AI system
5. **Interview** — Senior/staff engineer-level discussion questions

---

## Quiz 01: Foundations of AI

### Multiple Choice

**Q1.** Which of the following best describes the difference between supervised and unsupervised learning?

- [ ] A) Supervised learning requires GPUs; unsupervised does not
- [x] B) Supervised learning requires labelled training data; unsupervised learning does not
- [ ] C) Supervised learning uses neural networks; unsupervised uses decision trees
- [ ] D) Supervised learning trains faster than unsupervised

**Q2.** The Transformer architecture was introduced in which paper?

- [ ] A) "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky, 2012)
- [ ] B) "Deep Residual Learning for Image Recognition" (He, 2015)
- [x] C) "Attention Is All You Need" (Vaswani, 2017)
- [ ] D) "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin, 2018)

**Q3.** Which component of an enterprise AI system is responsible for retrieving relevant context before generation?

- [ ] A) The LLM
- [ ] B) The API Gateway
- [x] C) The Retrieval Layer / Vector Database
- [ ] D) The Business Logic Layer

---

## Quiz 02: Mathematics for AI

### Short Answer

**Q1.** Derive the gradient of the MSE loss \(L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2\) with respect to the weights \(w\), where \(\hat{y}_i = w^T x_i\).

??? answer
    \[
    \frac{\partial L}{\partial w} = -\frac{2}{n} X^T (y - \hat{y})
    \]

**Q2.** Explain in one paragraph why cross-entropy is preferred over MSE as a loss function for classification tasks.

??? answer
    MSE applied to classification outputs penalises prediction errors proportional to the square of the error, which can lead to vanishingly small gradients for confidently wrong predictions (when the sigmoid output is near 0 or 1). Cross-entropy \(-\sum_c y_c \log \hat{y}_c\) produces gradients proportional to the prediction error \(\hat{y} - y\), providing stronger, more consistent learning signal even when predictions are very wrong.

---

## Quiz 03: Machine Learning

### Coding

**Q1.** Implement k-fold cross-validation from scratch in NumPy (do not use sklearn's `KFold`).

```python
import numpy as np

def kfold_cv(X: np.ndarray, y: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns k (train_idx, val_idx) pairs.
    
    Example:
        >>> folds = kfold_cv(X, y, k=5)
        >>> for train_idx, val_idx in folds:
        ...     X_train, X_val = X[train_idx], X[val_idx]
    """
    ...
```

**Q2.** A model achieves 95% accuracy on a binary classification task. The dataset has 95% negative samples and 5% positive samples. What is the precision, recall, and F1 for the positive class if the model predicts "negative" for every sample?

??? answer
    - Precision: undefined (no positive predictions → TP=0, FP=0)
    - Recall: 0.0 (all positives are missed)
    - F1: 0.0
    - Lesson: accuracy is a misleading metric for imbalanced classes.

---

## Interview Question Bank

### Foundations

1. Explain the bias-variance tradeoff. How does adding more training data typically affect bias and variance?
2. When would you choose k-fold cross-validation over a held-out validation set?
3. Describe three sources of data leakage and how you would prevent each.

### Deep Learning

4. Why does batch normalisation help training? What happens at inference time?
5. Explain vanishing gradients. How do residual connections and LSTMs address this problem?
6. You observe training loss decreasing but validation loss increasing after epoch 10. What are three actions you would take?

### LLMs & RAG

7. A user reports that your RAG system is giving correct information but attributing it to the wrong sources. What do you investigate?
8. Explain the KV cache in LLMs. What is its memory cost and how does it change with sequence length?
9. When would you prefer fine-tuning over RAG? Give a concrete business scenario for each.

### System Design

10. Design a real-time content moderation system for user-generated text that handles 100,000 requests per minute. Discuss model choice, latency requirements, false positive/negative tradeoffs, and monitoring.
