# Two-Stage Cancer Classification Pipeline

A machine learning pipeline that classifies biological samples into five cancer stages using a two-stage sequential classification approach. Built to handle severe class imbalance across a high-dimensional (~350 feature) dataset.

---

## Problem Statement

Given a dataset of ~350 biological/genomic features per sample, the goal is to classify each sample into one of five categories:

- Healthy
- Screening stage cancer
- Early stage cancer
- Mid stage cancer
- Late stage cancer

A naive single-model approach fails here due to severe class imbalance — healthy samples are heavily underrepresented. A model trained directly on all five classes will learn to ignore the minority class entirely, producing misleading accuracy scores while missing the most clinically critical cases.

---

## Approach: Two-Stage Classification

To address this, the problem is decomposed into two sequential classification steps:

**Stage 1 — Coarse 4-class classification**

`healthy+screening` (combined) vs `early stage` vs `mid stage` vs `late stage`

Merging healthy and screening samples into a single class produces a more balanced 4-class problem. This stage handles the bulk of the classification work.

**Stage 2 — Fine-grained binary classification**

`healthy` vs `screening stage cancer`

Samples predicted as `healthy+screening` by Stage 1 are passed to a dedicated binary classifier that separates them. This is the most clinically sensitive distinction, and isolating it allows the model to focus entirely on this difficult boundary.

---

## Models

| Stage | Algorithm | Key Technique |
|-------|-----------|---------------|
| Stage 1 | Random Forest | `class_weight='balanced'`, GridSearchCV on weighted recall |
| Stage 1 | Neural Network (TensorFlow/Keras) | Class-weighted loss, Dropout, Early Stopping |
| Stage 2 | XGBoost | `scale_pos_weight` tuning via GridSearchCV, Stratified K-Fold |
| Stage 2 | Gaussian Naive Bayes | Probabilistic baseline for continuous high-dimensional features |

---

## Technical Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas, NumPy | Data loading, manipulation, and numerical operations |
| scikit-learn | Preprocessing, model training, GridSearchCV, evaluation metrics |
| TensorFlow / Keras | Neural network architecture and training |
| XGBoost | Gradient boosted tree classifier for imbalanced binary classification |
| Matplotlib | Class distribution visualisation and confusion matrix display |

---

## Project Structure

```
├── cancer_classification.ipynb   # Main notebook (cleaned, fully documented)
├── Train_Set.csv                 # Training data (~350 features, labelled)
├── Test_Set.csv                  # Held-out test data
└── README.md
```

---

## Key Design Decisions

**Class imbalance handling** — Rather than resampling (SMOTE, undersampling), class weighting is used throughout. Random Forest uses `class_weight='balanced'`, the Neural Network computes and applies class weights during training, and XGBoost's `scale_pos_weight` is tuned via grid search to penalise minority class misclassification.

**Preprocessing separation** — Stage 1 uses L2 row-wise normalisation (`sklearn.preprocessing.normalize`) appropriate for high-dimensional sparse-like data. Stage 2 uses `StandardScaler` to standardise features, fitted exclusively on training data and applied via `transform` on the test set to prevent data leakage.

**Hyperparameter tuning** — `GridSearchCV` with 5-fold cross-validation is used for both Random Forest (optimising weighted recall) and XGBoost (optimising F1 score). Weighted recall is chosen as the primary metric for Stage 1 because it reflects performance across all classes proportionally under imbalance.

**Stratified cross-validation** — `StratifiedKFold` is used for XGBoost tuning to ensure each fold maintains the original class ratio, which is essential when one class represents a very small fraction of the data.

---

## How to Run

1. Clone the repository and ensure `Train_Set.csv` and `Test_Set.csv` are in the root directory.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn tensorflow xgboost matplotlib
   ```
3. Open and run `cancer_classification.ipynb` top to bottom in Jupyter or Google Colab.

---

## References

- Akinnuwesi, B. A. (2022). Application of support vector machine algorithm for early differential diagnosis of prostate cancer. *Data Science and Management*.
- Khorshid, S. F., & Abdulazeez, A. M. (2021). Breast cancer diagnosis based on K-Nearest Neighbors: A Review. *PalArch's Journal of Archaeology of Egypt / Egyptology*.
