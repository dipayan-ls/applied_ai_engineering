# Chapter 2 — Supervised Learning

!!! abstract "Chapter Summary"
    Supervised learning is the workhorse of applied ML. This chapter covers the canonical algorithms — from simple linear regression to gradient-boosted trees — with full mathematical derivations, Python implementations, and concrete guidance on when each algorithm excels and where it fails.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Derive the normal equation for linear regression and explain when to prefer gradient descent over a closed-form solution.
2. Implement logistic regression from scratch using only NumPy, including the sigmoid, binary cross-entropy, and gradient update.
3. Explain how information gain drives decision tree splits and implement cost-complexity pruning.
4. Articulate the difference between bagging (Random Forests) and boosting (XGBoost/LightGBM) and their implications for bias–variance.
5. Design a principled hyperparameter search strategy using Bayesian optimisation with Optuna.

---

## 2.1 Linear Regression

### 2.1.1 The Hypothesis

Linear regression models the output as a weighted sum of inputs plus a bias:

$$
\hat{y} = w^\top x + b = \sum_{j=1}^{d} w_j x_j + b
$$

In matrix notation, absorbing the bias into the weight vector via an augmented feature vector $\tilde{x} = [1, x_1, \ldots, x_d]^\top$:

$$
\hat{y} = \tilde{w}^\top \tilde{x}, \quad \hat{Y} = X\tilde{w}
$$

where $X \in \mathbb{R}^{n \times (d+1)}$ has a leading column of ones.

### 2.1.2 Mean Squared Error Loss

$$
\mathcal{L}_{\text{MSE}}(w) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{n} \|Xw - y\|^2
$$

### 2.1.3 The Normal Equation

Setting $\nabla_w \mathcal{L} = 0$:

$$
\nabla_w \mathcal{L} = \frac{2}{n} X^\top (Xw - y) = 0 \implies X^\top X w = X^\top y
$$

$$
\boxed{w^* = (X^\top X)^{-1} X^\top y}
$$

### 2.1.4 Gradient Descent

When $n$ or $d$ is large, computing $(X^\top X)^{-1}$ is infeasible ($O(d^3)$ cost). Gradient descent iteratively updates:

$$
w \leftarrow w - \alpha \nabla_w \mathcal{L} = w - \frac{2\alpha}{n} X^\top (Xw - y)
$$

where $\alpha$ is the learning rate.

### 2.1.5 Closed-Form vs Iterative

| Criterion | Normal Equation | Gradient Descent |
|-----------|----------------|-----------------|
| Features $d$ | Works for $d < 10{,}000$ | Scales to millions |
| Samples $n$ | Works for $n < 1{,}000{,}000$ | Scales to billions (SGD) |
| Hyperparameters | None | Learning rate, batch size, scheduler |
| Implementation | 3 lines | More code; convergence monitoring |
| Invertibility | Fails if $X^\top X$ is singular | Regularisation handles this |

### 2.1.6 Ridge and Lasso Regularisation

**Ridge** ($\ell_2$): adds $\lambda \|w\|^2$ to the loss. Closed form: $w^* = (X^\top X + \lambda I)^{-1} X^\top y$.

**Lasso** ($\ell_1$): adds $\lambda \|w\|_1$. Produces *sparse* solutions (drives irrelevant weights to exactly zero).

$$
\mathcal{L}_{\text{Ridge}} = \|Xw - y\|^2 + \lambda \|w\|^2
$$

$$
\mathcal{L}_{\text{Lasso}} = \|Xw - y\|^2 + \lambda \|w\|_1
$$

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_linear_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, Pipeline]:
    """Fit OLS, Ridge, and Lasso regression pipelines."""
    models: dict[str, Pipeline] = {
        "ols": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]),
        "lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10_000))]),
    }
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        print(f"{name:>5}: fitted")
    return models
```

---

## 2.2 Logistic Regression

### 2.2.1 The Sigmoid Function

For binary classification, we need outputs in $[0, 1]$. The sigmoid maps any real number:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^\top x + b
$$

$$
P(y = 1 \mid x; w) = \sigma(w^\top x + b)
$$

### 2.2.2 Binary Cross-Entropy Loss

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
$$

The gradient with respect to $w$ simplifies elegantly:

$$
\nabla_w \mathcal{L} = \frac{1}{n} X^\top (\hat{y} - y)
$$

### 2.2.3 From-Scratch Python Implementation

```python
import numpy as np


class LogisticRegressionScratch:
    """Binary logistic regression implemented from first principles."""

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000) -> None:
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for epoch in range(self.n_epochs):
            z = X @ self.weights_ + self.bias_
            y_hat = self._sigmoid(z)
            error = y_hat - y

            dw = (X.T @ error) / n_samples
            db = error.mean()

            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db

            if epoch % 100 == 0:
                loss = self._bce_loss(y, y_hat)
                self.loss_history_.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 2.2.4 Scikit-learn Version

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
])
lr_pipeline.fit(X_train, y_train)
```

!!! note "Multiclass Extensions"
    Logistic regression extends to multiclass via one-vs-rest (OvR) or multinomial (softmax). Set `multi_class="multinomial"` and `solver="lbfgs"` in sklearn.

---

## 2.3 Decision Trees

### 2.3.1 Information Gain

Decision trees split data by asking yes/no questions. The best split maximises **information gain**:

$$
IG(D, A) = H(D) - \sum_{v \in \text{vals}(A)} \frac{|D_v|}{|D|} \cdot H(D_v)
$$

where $H(D)$ is the Shannon entropy of node $D$:

$$
H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

### 2.3.2 Gini Impurity

An alternative splitting criterion (used by sklearn's CART):

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

Gini is computationally cheaper (no logarithm) and tends to produce similar trees in practice.

### 2.3.3 Tree Depth and Pruning

A fully grown tree (no depth limit) memorises the training data. Two main remedies:

**Pre-pruning**: stop splitting when:
- `max_depth` is reached.
- Node has fewer than `min_samples_split` examples.
- Information gain falls below a threshold.

**Cost-complexity pruning** (post-pruning): grow a full tree, then iteratively remove the subtree whose removal minimally increases error, scaled by a complexity parameter $\alpha$:

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import numpy as np

# Find optimal alpha via cross-validation
dt = DecisionTreeClassifier(random_state=42)
path = dt.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[::5]   # sample every 5th alpha to save time

cv_scores = []
for alpha in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(tree, X_train, y_train, cv=5, scoring="f1_macro")
    cv_scores.append(scores.mean())

best_alpha = alphas[np.argmax(cv_scores)]
best_tree = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
best_tree.fit(X_train, y_train)
print(export_text(best_tree, feature_names=feature_names, max_depth=4))
```

---

## 2.4 Random Forests

### 2.4.1 Bagging

**Bootstrap aggregation (bagging)**: train $B$ trees, each on an independent bootstrap sample of the training set (sampling with replacement). Aggregate predictions by majority vote (classification) or mean (regression).

Each bootstrap sample contains approximately $63.2\%$ of unique training examples — the remaining $36.8\%$ form the **out-of-bag (OOB)** set for that tree.

### 2.4.2 Feature Subsampling

In addition to bagging, each split in a Random Forest considers only a random subset of $m$ features (typically $m = \sqrt{d}$ for classification, $m = d/3$ for regression). This *decorrelates* the trees, reducing variance beyond what bagging alone achieves.

### 2.4.3 Out-of-Bag Error

Each example is OOB for approximately $37\%$ of trees. We can compute predictions using only those trees, yielding an unbiased estimate of generalisation error **without a separate validation set**:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    oob_score=True,        # enable OOB estimate
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print(f"OOB accuracy: {rf.oob_score_:.4f}")
```

### 2.4.4 Feature Importance

Random Forests provide two importance measures:

**Mean Decrease in Impurity (MDI)**: average decrease in Gini/entropy across all splits on a feature. Fast but biased toward high-cardinality features.

**Permutation Importance**: measure accuracy drop when a feature's values are randomly shuffled. Model-agnostic and unbiased, but slower.

```python
from sklearn.inspection import permutation_importance
import pandas as pd

result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)
print(importance_df.head(10).to_string(index=False))
```

---

## 2.5 Gradient Boosting

### 2.5.1 The Additive Model

Gradient boosting builds an ensemble *sequentially*. The model after $M$ rounds is:

$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot h_m(x)
$$

where $\eta$ is the learning rate (shrinkage) and $h_m$ is a weak learner (usually a shallow tree) fit to the **negative gradient** of the loss:

$$
r_m^{(i)} = -\left[\frac{\partial \mathcal{L}(y^{(i)}, F(x^{(i)}))}{\partial F(x^{(i)})}\right]_{F = F_{m-1}}
$$

For MSE loss, $r_m^{(i)} = y^{(i)} - F_{m-1}(x^{(i)})$ — the residuals.

### 2.5.2 XGBoost, LightGBM, CatBoost Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Split strategy** | Level-wise | Leaf-wise | Symmetric trees |
| **Categorical features** | Manual encoding required | Native support (basic) | Best-in-class native support |
| **Speed (training)** | Fast | Fastest | Moderate |
| **Memory** | Moderate | Low | Moderate |
| **Regularisation** | L1, L2, tree complexity | L1, L2 | L2 + built-in bagging |
| **GPU support** | Yes (`device="cuda"`) | Yes | Yes |
| **Best for** | General-purpose benchmark | Large datasets | High-cardinality categoricals |
| **Key hyperparams** | `max_depth`, `eta`, `subsample` | `num_leaves`, `learning_rate` | `depth`, `learning_rate`, `l2_leaf_reg` |

```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
```

---

## 2.6 Support Vector Machines

### 2.6.1 The Maximum-Margin Hyperplane

A linear SVM finds the hyperplane $w^\top x + b = 0$ that maximises the margin $\frac{2}{\|w\|}$ between the two classes. The primal optimisation problem:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y^{(i)}(w^\top x^{(i)} + b) \geq 1, \; \forall i
$$

The **soft-margin** SVM (with slack variables $\xi_i$) allows misclassification:

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

The hyperparameter $C$ trades margin width against training error. Large $C$ = low bias, high variance; small $C$ = high bias, low variance.

### 2.6.2 The Kernel Trick

SVMs operate on dot products $x^{(i)\top} x^{(j)}$. Replacing with a kernel function $K(x^{(i)}, x^{(j)})$ implicitly maps inputs to a high-dimensional feature space without computing the mapping explicitly.

| Kernel | Formula | When to use |
|--------|---------|-------------|
| **Linear** | $x^\top z$ | Linearly separable, high-dimensional text data |
| **Polynomial** | $(x^\top z + c)^p$ | Moderate non-linearity, image features |
| **RBF (Gaussian)** | $\exp(-\gamma \|x - z\|^2)$ | General non-linear problems |
| **Sigmoid** | $\tanh(\kappa x^\top z + c)$ | Rare; use RBF instead |

### 2.6.3 Hyperparameter Guidance

- **$C$**: start at 1.0; search log-scale $[10^{-3}, 10^3]$.
- **$\gamma$** (RBF): `"scale"` (default: $1/(d \cdot \text{Var}(X))$) is a good starting point; search log-scale $[10^{-4}, 10^1]$.

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # SVMs require feature scaling
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
])
svm_pipeline.fit(X_train, y_train)
```

!!! warning "SVM Scaling Requirement"
    SVMs are **not** scale-invariant. Always standardise features before fitting an SVM. Failure to do so typically degrades performance dramatically.

---

## 2.7 k-Nearest Neighbours

### 2.7.1 The Algorithm

KNN makes predictions by finding the $k$ training examples closest to the query point and aggregating their labels:

$$
\hat{y} = \text{mode}\{y^{(j)} : j \in \text{kNN}(x)\} \quad \text{(classification)}
$$

$$
\hat{y} = \frac{1}{k} \sum_{j \in \text{kNN}(x)} y^{(j)} \quad \text{(regression)}
$$

### 2.7.2 Distance Metrics

| Metric | Formula | Use case |
|--------|---------|----------|
| **Euclidean** ($L^2$) | $\sqrt{\sum_j (x_j - z_j)^2}$ | Continuous features, equal scale |
| **Manhattan** ($L^1$) | $\sum_j |x_j - z_j|$ | High-dimensional, robust to outliers |
| **Minkowski** ($L^p$) | $\left(\sum_j |x_j - z_j|^p\right)^{1/p}$ | Generalises both |
| **Cosine** | $1 - \frac{x^\top z}{\|x\|\|z\|}$ | Text, sparse features |
| **Hamming** | Fraction of differing positions | Categorical / binary features |

### 2.7.3 The Curse of Dimensionality

In high dimensions, the concept of "nearness" breaks down:

- The volume of a hypersphere shrinks relative to the enclosing hypercube.
- All pairwise distances converge to the same value: $\frac{\max d - \min d}{\min d} \to 0$ as $d \to \infty$.
- A dataset that densely covers $[0,1]^2$ with 100 points requires $100^{10}$ points to achieve the same coverage in $[0,1]^{10}$.

!!! tip "KNN Best Practices"
    - Always standardise features before computing distances.
    - Use `KNeighborsClassifier(algorithm="ball_tree")` for $d < 20$, `"brute"` for sparse data.
    - Tune $k$ via cross-validation; odd values avoid ties.
    - KNN is strong for $d < 20$ and $n < 100{,}000$; use other algorithms beyond those thresholds.

---

## 2.8 Hyperparameter Tuning

### 2.8.1 Grid Search

Exhaustively evaluates all combinations in a specified grid. Guaranteed to find the best configuration within the grid, but scales as the product of all values:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__gamma": ["scale", 0.001, 0.01, 0.1],
}
grid_search = GridSearchCV(
    svm_pipeline, param_grid,
    cv=5, scoring="f1_macro", n_jobs=-1, verbose=2,
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")
```

### 2.8.2 Random Search

Samples $n$ random configurations. Bergstra & Bengio (2012) showed random search is more efficient than grid search because most hyperparameters are relatively unimportant — random search can explore wider ranges efficiently.

### 2.8.3 Bayesian Optimisation with Optuna

Bayesian optimisation builds a probabilistic *surrogate model* of the objective function and uses it to select the next configuration to evaluate (the acquisition function). Optuna uses the Tree-structured Parzen Estimator (TPE):

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: returns mean CV F1 score to maximise."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="f1_macro", n_jobs=-1,
    )
    return float(scores.mean())


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, timeout=600, show_progress_bar=True)

print(f"Best trial value: {study.best_value:.4f}")
print(f"Best params:      {study.best_params}")
```

### 2.8.4 Search Strategy Comparison

| Strategy | Pros | Cons | Recommended for |
|----------|------|------|----------------|
| **Grid Search** | Exhaustive, reproducible | Combinatorial explosion | ≤ 3 params, small grid |
| **Random Search** | Simple, parallelisable | No learning from history | 3–6 params, tight budget |
| **Bayesian (Optuna/SMAC)** | Learns which regions are promising | More complex setup | > 4 params, expensive model |
| **Successive Halving** | Early-stops poor configs | Requires holdout budget | Large grid, fast training |

---

## 2.9 Exercises

!!! question "Exercise 2.1 — Normal Equation"
    Derive the normal equation starting from $\mathcal{L}(w) = \frac{1}{n}\|Xw - y\|^2$. Show all steps. Under what conditions does $(X^\top X)^{-1}$ not exist, and how does Ridge regression fix this?

!!! question "Exercise 2.2 — Logistic Regression from Scratch"
    Extend the `LogisticRegressionScratch` class to:

    a. Support L2 regularisation (Ridge), adding $\frac{\lambda}{2}\|w\|^2$ to the loss.
    b. Support mini-batch gradient descent (batch size as a hyperparameter).
    c. Plot the training loss curve.

    Verify on the `breast_cancer` dataset that your implementation matches `sklearn.LogisticRegression`.

!!! question "Exercise 2.3 — Decision Tree vs Random Forest"
    On the `wine` dataset:

    a. Train a full decision tree and report training and CV accuracy.
    b. Vary `max_depth` from 1 to 20 and plot training vs. validation accuracy.
    c. Train a Random Forest with 300 trees and compare OOB accuracy with CV accuracy.
    d. Explain, in bias–variance terms, why the Random Forest outperforms a deep single tree.

!!! question "Exercise 2.4 — Gradient Boosting Hyperparameter Interaction"
    Using LightGBM on the `california_housing` dataset:

    a. Show that decreasing `learning_rate` without increasing `n_estimators` harms performance.
    b. Use Optuna to jointly tune `learning_rate`, `n_estimators`, and `num_leaves`.
    c. Compare results with a default Random Forest.

!!! question "Exercise 2.5 — SVM Kernel Comparison"
    On the `digits` dataset (8×8 grayscale images):

    a. Train a linear SVM and an RBF SVM and compare 5-fold CV accuracy.
    b. Grid-search over $C \in [0.01, 100]$ and $\gamma \in [10^{-4}, 10^{-1}]$ for the RBF SVM.
    c. Visualise the effect of $C$ and $\gamma$ on the validation score using a heatmap.
    d. Explain geometrically why the RBF kernel is better suited to this problem.

---

## 2.10 Seminal Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|-------------|
| *A Training Algorithm for Optimal Margin Classifiers* | Boser, Guyon, Vapnik | 1992 | Introduced the original SVM with the kernel trick |
| *Random Forests* | Breiman | 2001 | Defined the random forest algorithm; proved OOB error bound |
| *Greedy Function Approximation: A Gradient Boosting Machine* | Friedman | 2001 | Unified boosting as gradient descent in function space |
| *XGBoost: A Scalable Tree Boosting System* | Chen & Guestrin | 2016 | Introduced regularised boosting, hardware-aware optimisation |
| *Random Search for Hyper-Parameter Optimization* | Bergstra & Bengio | 2012 | Proved random search dominates grid search for most problems |

---

## Summary

| Algorithm | Inductive Bias | Scalability | Interpretability | Regularisation |
|-----------|---------------|-------------|-----------------|----------------|
| Linear Regression | Linearity | $O(nd)$ / $O(d^3)$ | High | Ridge, Lasso |
| Logistic Regression | Linearity in log-odds | $O(nd)$ | High | L1, L2 |
| Decision Tree | Axis-aligned splits | $O(nd \log n)$ | Very high | Pruning, depth |
| Random Forest | Ensemble of trees | $O(B \cdot nd\sqrt{d} \log n)$ | Medium (feature importance) | None needed |
| Gradient Boosting | Sequential residual correction | $O(B \cdot nd\log n)$ | Low | Shrinkage, tree params |
| SVM | Maximum margin | $O(n^2 d)$ to $O(n^3)$ | Low | $C$ |
| KNN | Local smoothness | $O(nd)$ at inference | High | $k$ |

---

*Next: [Chapter 3 — Unsupervised Learning](../ch03-unsupervised/index.md)*
