---
title: "Ch 4 — Model Evaluation & Metrics"
---

# Ch 4 — Model Evaluation & Metrics

<div class="chapter-meta" markdown>
| | |
|---|---|
| **Difficulty** | Intermediate |
| **Reading time** | 75 min |
| **Prerequisites** | Ch 1 (ML Fundamentals), Ch 2 (Supervised Learning), Ch 3 (Unsupervised Learning) |
</div>

---

## Learning Objectives

By the end of this chapter you will be able to:

1. **Construct and interpret** a confusion matrix and derive Precision, Recall, F1-score, Specificity, and AUC-ROC from its four cells.
2. **Select the appropriate metric** for a given task — distinguishing when F1, AUC-PR, or RMSE is the right choice based on class imbalance and cost asymmetry.
3. **Implement** ROC curves, Precision-Recall curves, and calibration reliability diagrams using `scikit-learn`.
4. **Apply statistical tests** (paired t-test, McNemar's test) to determine whether the difference between two models is statistically significant rather than due to sampling noise.
5. **Monitor deployed models** using Population Stability Index (PSI) for data drift and identify the signs and remedies for concept drift in production systems.

---

## 4.1 Why Evaluation Matters More Than Accuracy Alone

Consider a medical screening model for a rare cancer affecting 1 % of the population. A model that always predicts *"no cancer"* achieves **99 % accuracy** — yet it is completely useless. It catches zero true cases and provides a dangerous false sense of security.

This thought experiment reveals the central tension in machine learning evaluation: **accuracy is a deceptive metric when classes are imbalanced, when prediction errors have asymmetric costs, or when the decision threshold matters**.

Robust evaluation requires a portfolio of metrics, each designed to capture a different facet of model behaviour:

| Concern | Metric(s) to use |
|---|---|
| Overall correctness | Accuracy |
| Cost of false positives | Precision |
| Cost of false negatives | Recall |
| Balance of both error types | F1-score |
| Threshold-independent ranking | AUC-ROC |
| Rare-event performance | AUC-PR (Average Precision) |
| Continuous output error | MAE, RMSE, R² |
| Confidence reliability | Calibration / Brier score |
| Deployment distribution shift | PSI, concept drift metrics |

!!! warning "The accuracy trap"
    Never report accuracy alone on imbalanced datasets. Always check the class distribution first with `y.value_counts()`. If the majority class exceeds 80 % of samples, accuracy is almost meaningless as a standalone metric — a naive majority-class classifier would beat most models on this metric alone.

---

## 4.2 Classification Metrics from the Confusion Matrix

### 4.2.1 The Confusion Matrix

For a binary classifier, every prediction falls into exactly one of four categories:

|  | Predicted Positive | Predicted Negative |
|---|:---:|:---:|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

From these four values, every standard classification metric is derived. They can be read as: TP = correctly predicted positive; FN = missed positive (Type II error); FP = false alarm (Type I error); TN = correctly predicted negative.

### 4.2.2 Core Derived Metrics

**Accuracy** — overall fraction correct:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** (Positive Predictive Value) — of all *predicted* positives, how many are actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (Sensitivity, True Positive Rate) — of all *actual* positives, how many did the model catch?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Specificity** (True Negative Rate) — of all actual negatives, how many were correctly identified?

$$\text{Specificity} = \frac{TN}{TN + FP} = 1 - \text{FPR}$$

**F1-Score** — harmonic mean of Precision and Recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

The harmonic mean is used because it penalises extreme imbalance between Precision and Recall. A model with Precision = 1.0 and Recall = 0.01 has F1 ≈ 0.02, not 0.505. It forces both to be high simultaneously.

**Generalised F-beta** — allows weighting Recall $\beta$ times more than Precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta = 2$: Recall weighted twice as heavily — medical diagnosis where missing a disease is catastrophic.
- $\beta = 0.5$: Precision weighted twice as heavily — spam filtering where false positives annoy users.

### 4.2.3 Worked Numeric Example

A fraud detection model is evaluated on 10,000 transactions:

- TP = 80 (frauds correctly caught)
- FN = 20 (frauds missed — the model says legitimate)
- FP = 100 (legitimate transactions flagged as fraud — false alarms)
- TN = 9,800 (legitimate transactions correctly cleared)

Step-by-step derivation:

$$\text{Accuracy} = \frac{80 + 9800}{10000} = \frac{9880}{10000} = 98.8\%$$

$$\text{Precision} = \frac{80}{80 + 100} = \frac{80}{180} \approx 44.4\%$$

$$\text{Recall} = \frac{80}{80 + 20} = \frac{80}{100} = 80.0\%$$

$$\text{Specificity} = \frac{9800}{9800 + 100} = \frac{9800}{9900} \approx 99.0\%$$

$$F_1 = 2 \cdot \frac{0.444 \times 0.800}{0.444 + 0.800} = \frac{0.711}{1.244} \approx 57.1\%$$

!!! note "Interpretation"
    Accuracy of 98.8 % sounds impressive, but the model misses 20 % of fraud cases and raises 100 false alarms for every 80 genuine catches. Whether this is acceptable depends on the business cost: if a missed fraud costs £5,000 and a false positive costs £10 (manual review), the expected daily cost at these rates is £2,000 missed + £1,000 reviews = £3,000. Recall is the priority metric, not accuracy.

=== "Python"

    ```python
    from __future__ import annotations

    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    # Ground truth: 100 frauds, 9900 legitimate
    y_true = np.array([1] * 100 + [0] * 9900)

    # Simulated predictions: 80 TP, 20 FN, 100 FP, 9800 TN
    y_pred = np.array(
        [1] * 80 + [0] * 20          # 80 TP, 20 FN (out of 100 positives)
        + [1] * 100 + [0] * 9800     # 100 FP, 9800 TN (out of 9900 negatives)
    )

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    # [[9800  100]
    #  [  20   80]]  ← rows = actual, cols = predicted

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Full report including macro, micro, and weighted averages
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    ```

---

## 4.3 The ROC Curve and AUC-ROC

### 4.3.1 What Is a ROC Curve?

The **Receiver Operating Characteristic (ROC) curve** plots the True Positive Rate (Recall) on the y-axis against the False Positive Rate (1 − Specificity) on the x-axis as the **decision threshold** is swept from 1.0 down to 0.0.

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

At threshold = 1.0: the model predicts nothing positive → TPR = 0, FPR = 0 (origin).  
At threshold = 0.0: the model predicts everything positive → TPR = 1, FPR = 1 (top-right corner).

A **perfect classifier** has a curve that jumps straight to (0, 1) — AUC = 1.0.  
A **random classifier** follows the diagonal — AUC = 0.5.

### 4.3.2 AUC Interpretation

The **Area Under the ROC Curve (AUC-ROC)** has a clean probabilistic interpretation:

> AUC = the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example.

| AUC Range | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent discrimination |
| 0.80 – 0.90 | Good |
| 0.70 – 0.80 | Fair |
| 0.60 – 0.70 | Poor |
| 0.50 – 0.60 | Fail (near-random) |

AUC-ROC is **threshold-independent**, making it suitable for comparing classifiers without committing to an operating point. However, it can be overly optimistic on imbalanced datasets — see Section 4.4.

=== "Python"

    ```python
    from __future__ import annotations

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import train_test_split

    # Imbalanced binary dataset: 10 % positive class
    X, y = make_classification(
        n_samples=5_000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()

    print(f"AUC-ROC: {roc_auc:.4f}")
    ```

---

## 4.4 Precision-Recall Curve and Average Precision

### 4.4.1 When to Prefer PR over ROC

ROC curves can be **overly optimistic on imbalanced datasets** because the FPR denominator (FP + TN) is dominated by the large pool of true negatives. Even a model that generates thousands of false positives appears to have a low FPR when there are millions of true negatives.

The **Precision-Recall curve** directly tracks the trade-off between Precision and Recall without TN anywhere in the formula — making it the correct choice for:

- Rare event detection: fraud, disease, manufacturing defects
- Information retrieval and document ranking
- Any task where the positive class is the minority and its performance is what matters

### 4.4.2 Average Precision (AP)

The **Average Precision** summarises the area under the PR curve as a weighted mean of precision values at each threshold:

$$\text{AP} = \sum_{k} (R_k - R_{k-1}) \cdot P_k$$

This equals the area under the PR curve (AUC-PR). The **random baseline** for AP is the positive class prevalence: if 5 % of samples are positive, a random classifier has AP ≈ 0.05.

=== "Python"

    ```python
    from __future__ import annotations

    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
    )

    # y_test and y_scores from the ROC example above
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(recall, precision, lw=2, where="post",
            label=f"Logistic Regression (AP = {ap:.3f})")
    ax.axhline(
        y=y_test.mean(),
        color="k",
        linestyle="--",
        lw=1,
        label=f"Random baseline (prevalence = {y_test.mean():.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=150)
    plt.show()

    print(f"Average Precision (AUC-PR): {ap:.4f}")
    ```

!!! tip "Choosing the operating point"
    Once you have your PR curve, choose the threshold that satisfies your business constraint. For example: *"We can review at most 500 fraud alerts per day"* — find the threshold where your model produces ≤ 500 positives per day and read off Precision and Recall at that point.

---

## 4.5 Regression Metrics

For tasks with a continuous numeric target, the confusion-matrix framework does not apply. The following metrics measure the magnitude and direction of prediction errors. Let $\hat{y}_i$ be the model's prediction, $y_i$ the true value, $\bar{y}$ the mean of the true values, and $n$ the total sample count.

### 4.5.1 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Same units as the target variable — directly interpretable.
- Robust to outliers (linear penalty per error).
- **Use when**: the target has a symmetric error distribution and outliers should not dominate.

### 4.5.2 Mean Squared Error (MSE) and Root MSE

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}$$

- MSE penalises large errors quadratically: a single error of 10 contributes as much as 100 errors of 1.
- RMSE restores the original units, making it more interpretable than MSE.
- **Use when**: large errors are disproportionately costly (structural engineering, financial forecasting).

### 4.5.3 Coefficient of Determination (R²)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$

- Proportion of variance in $y$ explained by the model relative to a naive mean predictor.
- $R^2 = 1$: perfect; $R^2 = 0$: as good as predicting $\bar{y}$; $R^2 < 0$: worse than predicting $\bar{y}$.
- **Caution**: $R^2$ increases monotonically as features are added (use adjusted $R^2$ for model comparison).

### 4.5.4 Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- Scale-independent — useful for comparing across datasets with different units (e.g., housing prices vs. electricity consumption).
- **Undefined when $y_i = 0$**; numerically unstable when $y_i \approx 0$.
- **Use when**: relative percentage error matters (demand forecasting, revenue prediction).

=== "Python"

    ```python
    from __future__ import annotations

    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler


    def mean_absolute_percentage_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute MAPE, guarding against division by zero."""
        mask = y_true != 0
        return float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )


    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"MAE  : {mae:.4f}  (in units of $100k)")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    ```

---

## 4.6 Multi-Class Metrics: Micro, Macro, and Weighted Averaging

When there are $K > 2$ classes, precision, recall, and F1 must be aggregated across classes. Three standard strategies exist:

### 4.6.1 Macro Averaging

Compute the metric **independently for each class** and take the unweighted mean:

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

- Treats every class as equally important regardless of support (number of samples).
- Sensitive to performance on rare classes — a model that fails on a minority class is penalised heavily.
- **Use when**: each class is equally important regardless of frequency (e.g., multiclass medical diagnosis).

### 4.6.2 Micro Averaging

Aggregate TP, FP, and FN counts globally across all classes, then compute the metric once:

$$\text{Micro-Precision} = \frac{\sum_{k=1}^K TP_k}{\sum_{k=1}^K (TP_k + FP_k)}$$

- Dominated by performance on frequent classes.
- For balanced datasets, Micro-F1 equals Accuracy.
- **Use when**: overall sample-level correctness matters regardless of class.

### 4.6.3 Weighted Averaging

Weigh each class metric by its support (number of true instances):

$$\text{Weighted-F1} = \frac{\sum_{k=1}^K \text{support}_k \cdot F1_k}{\sum_{k=1}^K \text{support}_k}$$

- Balances class importance against frequency — a middle ground between macro and micro.
- **Use when**: class imbalance exists and common classes should carry more weight, but rare classes still matter.

=== "Python"

    ```python
    from __future__ import annotations

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Per-class and aggregate metrics
    print(classification_report(
        y_test, y_pred,
        target_names=["setosa", "versicolor", "virginica"],
    ))

    # Individual averages
    for avg in ("macro", "micro", "weighted"):
        score = f1_score(y_test, y_pred, average=avg)
        print(f"F1 ({avg:>8}): {score:.4f}")
    ```

---

## 4.7 Statistical Significance of Model Comparisons

Training two models and comparing mean test scores is insufficient — any difference could be due to the random data split. Statistical tests quantify how confident we can be that one model genuinely outperforms another.

### 4.7.1 5×2 Cross-Validation Paired t-Test

Dietterich (1998) recommends the **5×2 cross-validation paired t-test** as the gold standard for classifier comparison:

1. Repeat 5 times: randomly split the dataset 50/50 into two folds.
2. Train each model on fold 1, evaluate on fold 2, and vice versa.
3. Record the per-fold difference $d_i^{(j)} = \text{err}_A^{(j)} - \text{err}_B^{(j)}$ for repeat $i$, fold $j$.
4. Compute fold means: $\bar{d}_i = (d_i^{(1)} + d_i^{(2)})/2$
5. Compute fold variances: $s_i^2 = (d_i^{(1)} - \bar{d}_i)^2 + (d_i^{(2)} - \bar{d}_i)^2$

Test statistic:

$$t = \frac{d_1^{(1)}}{\sqrt{\frac{1}{5}\sum_{i=1}^5 s_i^2}}$$

Under H₀ (equal performance), $t$ follows a $t$-distribution with 5 degrees of freedom.

### 4.7.2 McNemar's Test

For comparing two classifiers on a **single fixed test set**, count how often the models disagree:

|  | Model B correct | Model B incorrect |
|---|:---:|:---:|
| **Model A correct** | $n_{11}$ | $n_{10}$ |
| **Model A incorrect** | $n_{01}$ | $n_{00}$ |

Only the *discordant* pairs ($n_{01}$ and $n_{10}$) are informative. The test statistic with continuity correction:

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

Under H₀, $\chi^2$ follows a chi-squared distribution with 1 degree of freedom. Reject H₀ (i.e., conclude the models differ) when $\chi^2 > 3.841$ (the 5 % critical value).

=== "Python"

    ```python
    from __future__ import annotations

    import numpy as np
    from scipy.stats import chi2


    def mcnemar_test(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
    ) -> tuple[float, float]:
        """
        McNemar's test for paired classifier comparison.

        Parameters
        ----------
        y_true   : True class labels.
        y_pred_a : Predictions from model A.
        y_pred_b : Predictions from model B.

        Returns
        -------
        statistic : Chi-squared test statistic (with continuity correction).
        p_value   : Two-sided p-value.
        """
        correct_a = y_pred_a == y_true
        correct_b = y_pred_b == y_true

        n01 = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
        n10 = int(np.sum(~correct_a & correct_b))   # A wrong, B correct

        if (n01 + n10) == 0:
            return 0.0, 1.0   # Models make identical errors

        statistic = float((abs(n01 - n10) - 1) ** 2 / (n01 + n10))
        p_value   = float(1 - chi2.cdf(statistic, df=1))
        return statistic, p_value


    # Demonstration
    rng = np.random.default_rng(42)
    n = 1_000
    y_true   = rng.integers(0, 2, size=n)
    # Model A: ~70% accuracy; Model B: ~75% accuracy
    y_pred_a = (rng.random(n) > 0.30).astype(int)
    y_pred_b = (rng.random(n) > 0.25).astype(int)

    stat, p = mcnemar_test(y_true, y_pred_a, y_pred_b)
    print(f"McNemar statistic : {stat:.4f}")
    print(f"p-value           : {p:.4f}")
    alpha = 0.05
    if p < alpha:
        print(f"Models differ significantly at α = {alpha}")
    else:
        print(f"No significant difference detected at α = {alpha}")
    ```

---

## 4.8 Calibration: Are Probabilities Trustworthy?

A well-calibrated model's stated probabilities match empirical frequencies. If it predicts 80 % probability of fraud, fraud should actually occur about 80 % of the time among those predictions.

### 4.8.1 Reliability Diagrams

A **reliability diagram** (calibration curve) plots, for each probability bin:

- **x-axis**: mean predicted probability in the bin (e.g., [0.7, 0.8))
- **y-axis**: actual fraction of positives in that bin

A perfectly calibrated model follows the diagonal. Deviations indicate:
- **Over-confident**: curve falls below the diagonal (predicts 0.9 but only 70 % are positive)
- **Under-confident**: curve rises above the diagonal (predicts 0.4 but 65 % are positive)

### 4.8.2 Brier Score

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

Lower is better (0 = perfect). A random classifier on balanced data has Brier ≈ 0.25. The Brier score jointly penalises poor calibration and poor discrimination.

### 4.8.3 Platt Scaling

Random Forests and SVMs often produce poorly-calibrated raw probabilities. **Platt scaling** fits a logistic regression on top of the raw model scores:

$$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

Parameters $A$ and $B$ are found by maximum likelihood on a held-out calibration fold. `sklearn` implements this and isotonic regression calibration via `CalibratedClassifierCV`.

=== "Python"

    ```python
    from __future__ import annotations

    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=3_000, n_features=20, n_informative=8, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    # Uncalibrated Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_train, y_train)
    probs_uncal = rf.predict_proba(X_test)[:, 1]

    # Platt scaling (sigmoid calibration)
    rf_platt = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, random_state=0),
        method="sigmoid",
        cv=5,
    )
    rf_platt.fit(X_train, y_train)
    probs_platt = rf_platt.predict_proba(X_test)[:, 1]

    # Isotonic regression calibration
    rf_iso = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, random_state=0),
        method="isotonic",
        cv=5,
    )
    rf_iso.fit(X_train, y_train)
    probs_iso = rf_iso.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in [
        ("Uncalibrated RF", probs_uncal),
        ("Platt scaling",   probs_platt),
        ("Isotonic",        probs_iso),
    ]:
        CalibrationDisplay.from_predictions(
            y_test, probs, n_bins=10, ax=ax, name=name
        )
    ax.set_title("Calibration Curves (Reliability Diagrams)")
    plt.tight_layout()
    plt.savefig("calibration.png", dpi=150)
    plt.show()

    for name, probs in [
        ("Uncalibrated", probs_uncal),
        ("Platt scaling", probs_platt),
        ("Isotonic",      probs_iso),
    ]:
        brier = brier_score_loss(y_test, probs)
        print(f"Brier [{name:>14}]: {brier:.4f}")
    ```

---

## 4.9 Production Monitoring: Drift Detection

A model that scores well on a held-out test set may degrade once deployed because the real-world data distribution shifts over time. There are two primary types of drift:

### 4.9.1 Data Drift (Covariate Shift)

The input feature distribution $P(X)$ changes but the conditional relationship $P(Y|X)$ remains the same. Example: a model trained on summer purchase patterns is deployed during winter. The features (browsing categories, device type) shift in distribution, even though the underlying relationship between features and purchase intent hasn't changed.

**Population Stability Index (PSI)** quantifies the shift in a feature's distribution between the reference (training) and current (production) periods:

$$\text{PSI} = \sum_{i=1}^{B} \left( A_i - E_i \right) \cdot \ln\!\left(\frac{A_i}{E_i}\right)$$

where the sum is over $B$ bins, $E_i$ = expected (reference) proportion in bin $i$, $A_i$ = actual (current) proportion in bin $i$.

| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | No action needed |
| 0.10 – 0.25 | Moderate shift | Monitor closely, investigate features |
| > 0.25 | Significant shift | Retrain model or update features |

### 4.9.2 Concept Drift

The relationship $P(Y|X)$ itself changes. Example: customer credit risk behaviour shifts due to an economic recession, so the same financial features now predict different default probabilities. Concept drift requires retraining — no data transformation can repair a model that learned an obsolete relationship.

**Detection strategies**:

- Monitor the **prediction distribution**: if the proportion of high-confidence fraud predictions drops suddenly, something has changed.
- Track **business KPIs** tied to model outputs.
- Use **ADWIN** (Adaptive Windowing) or **Page-Hinkley** tests for online drift detection on streaming data.
- When ground truth labels arrive (even with delay), track rolling accuracy, F1, or AUC over time windows.

=== "Python"

    ```python
    from __future__ import annotations

    import numpy as np
    import pandas as pd


    def compute_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Compute the Population Stability Index (PSI).

        Parameters
        ----------
        expected : Reference distribution (e.g., training set model scores).
        actual   : Current distribution (e.g., production model scores).
        n_bins   : Number of quantile-based bins from the reference distribution.
        epsilon  : Small constant added to avoid log(0).

        Returns
        -------
        psi : PSI value.
        """
        # Define bin edges from the reference distribution quantiles
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(expected, quantiles)
        bin_edges[0]  -= 1e-8   # ensure leftmost bin captures minimum
        bin_edges[-1] += 1e-8   # ensure rightmost bin captures maximum

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts,   _ = np.histogram(actual,   bins=bin_edges)

        # Convert to proportions with epsilon smoothing
        e_pct = expected_counts / len(expected) + epsilon
        a_pct = actual_counts   / len(actual)   + epsilon

        bin_psi = (a_pct - e_pct) * np.log(a_pct / e_pct)
        return float(bin_psi.sum())


    def drift_report(
        train_scores: np.ndarray,
        prod_scores: np.ndarray,
        feature_name: str = "model_score",
    ) -> pd.DataFrame:
        """Generate a PSI drift report for a single feature."""
        psi = compute_psi(train_scores, prod_scores)
        status = (
            "OK"    if psi < 0.10 else
            "Watch" if psi < 0.25 else
            "Alert"
        )
        return pd.DataFrame([{
            "feature": feature_name,
            "psi":     round(psi, 4),
            "status":  status,
        }])


    rng = np.random.default_rng(0)

    train_scores = rng.normal(0.0, 1.0, 10_000)
    prod_moderate = rng.normal(0.3, 1.1, 5_000)   # mild shift
    prod_large    = rng.normal(1.5, 1.5, 5_000)   # significant shift

    for label, prod in [("moderate", prod_moderate), ("large", prod_large)]:
        report = drift_report(train_scores, prod, feature_name=f"score_{label}")
        print(report.to_string(index=False))
    ```

!!! warning "Ground truth latency"
    In production, true labels often arrive with delay — whether a loan defaults takes months to confirm. Monitor prediction and feature distributions immediately; wait for labels to track actual model accuracy. Use PSI as your early-warning system.

---

## 4.10 Exercises

**Exercise 4.1 — Threshold Tuning**  
A binary classifier has the following confusion matrix at threshold = 0.5: TP = 120, FP = 80, FN = 30, TN = 770. (a) Compute Precision, Recall, F1, and Specificity. (b) If False Negatives cost 10× more than False Positives, derive the expected daily cost at the current operating point and calculate whether lowering the threshold to reduce FN (accepting more FP) would reduce total cost, assuming FP increases to 160 and FN decreases to 10 when the threshold is lowered to 0.3.

**Exercise 4.2 — ROC vs PR Curves**  
Generate a synthetic dataset with 5 % positive class using `make_classification`. Train a `LogisticRegression` and a `DecisionTreeClassifier`. Plot their ROC curves and PR curves on the same figures. Explain in writing why the ROC curves appear more similar than the PR curves, and which metric gives a more honest picture of relative performance on this imbalanced task.

**Exercise 4.3 — Regression Metrics Sensitivity**  
Create two sets of predictions for 1,000 samples: one with uniformly distributed errors $\mathcal{U}(-2, 2)$, and one with a single extreme outlier ($\hat{y}_1 = y_1 + 1000$) but otherwise perfect predictions. Compute MAE, RMSE, and R² for each. Quantify the ratio RMSE/MAE for each set and explain how it diagnoses outlier sensitivity.

**Exercise 4.4 — Calibration Audit**  
Train an SVM (`SVC(probability=True)`) on the Breast Cancer Wisconsin dataset. Plot its reliability diagram before and after Platt scaling using `CalibratedClassifierCV`. Compute the Brier score for both. Report the relative improvement in calibration and explain why SVMs are prone to poor calibration.

**Exercise 4.5 — PSI Monitoring Pipeline**  
Implement a function `drift_report(train_df, prod_df, feature_cols)` that: (a) computes PSI for each numeric feature column, (b) flags features with PSI ∈ [0.1, 0.25) as "Watch" and PSI ≥ 0.25 as "Alert", and (c) returns a DataFrame sorted by PSI descending. Test it on two DataFrames where you deliberately engineer drift in a subset of features.

---

## Summary

### Key Formulas

| Metric | Formula | Range |
|---|---|---|
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | [0, 1] |
| Precision | $TP/(TP+FP)$ | [0, 1] |
| Recall | $TP/(TP+FN)$ | [0, 1] |
| Specificity | $TN/(TN+FP)$ | [0, 1] |
| F1-score | $2PR/(P+R)$ | [0, 1] |
| F-beta | $(1+\beta^2) \cdot PR / (\beta^2 P + R)$ | [0, 1] |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | [0, ∞) |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | [0, ∞) |
| R² | $1 - SS_{\text{res}}/SS_{\text{tot}}$ | (−∞, 1] |
| MAPE | $\frac{100}{n}\sum\|y_i-\hat{y}_i\|/y_i$ | [0, ∞) |
| Brier Score | $\frac{1}{n}\sum(\hat{p}_i-y_i)^2$ | [0, 1] |
| PSI | $\sum(A_i-E_i)\ln(A_i/E_i)$ | [0, ∞) |

### Key Takeaways

- Accuracy is insufficient for imbalanced datasets — always report at minimum Precision, Recall, and F1 together.
- ROC-AUC measures threshold-independent discrimination; AUC-PR better reflects rare-event performance.
- Statistical significance tests (McNemar, 5×2 paired t-test) are mandatory when comparing model performance.
- Calibration ensures predicted probabilities are reliable for business decision-making; Platt scaling or isotonic regression can correct miscalibrated models.
- Production monitoring requires tracking both data drift (PSI on feature and score distributions) and concept drift (label-based accuracy when labels are available).

!!! tip "Next steps"
    Proceed to Volume 4, Chapter 1 to see how these evaluation principles apply to neural networks, where the loss landscape, regularisation choices, and the training/validation/test split discipline have profound effects on the metrics discussed here.
