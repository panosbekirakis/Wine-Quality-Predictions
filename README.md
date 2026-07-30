# Wine Quality Prediction 🍷

## Overview

Binary classification of wine quality (**Good vs Not Good**) using supervised machine learning. The project combines the red and white *Wine Quality* datasets (UCI Machine Learning Repository) and compares two methods — **Support Vector Machines (SVM)** and **Decision Trees** — across training, evaluation, and hyperparameter tuning.

## Models

- SVM — Linear kernel
- SVM — RBF kernel
- Decision Tree (different depths)
- Random Forest (bonus, cross-validation only)

## Approach

- **Dataset:** red + white wines combined → 6,497 samples, 12 features (11 physicochemical + `color`)
- **Target:** binary `good = quality >= 6` → imbalanced classes (63% Good / 37% Not Good)
- **Split:** 80/20 stratified, with `StandardScaler` applied *after* the split (leakage-safe) inside a Pipeline
- **Validation:** 5-fold Stratified Cross-Validation
- **Tuning:** Grid Search **and** Random Search (C, gamma, max_depth, min_samples_leaf, criterion)

## Dataset

📁 Wine Quality — UCI Machine Learning Repository (red + white combined)

6,497 wine samples · 12 features · binary target (`good`)

## Evaluation Metrics

Accuracy · Precision · Recall · F1-score · ROC curve · AUC · Confusion Matrix

*Because the classes are imbalanced (63/37), **F1 and AUC** are treated as more reliable than plain accuracy.*

## Results (Test Set)

### Untuned models

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| SVM (linear) | 0.7346 | 0.8009 | 0.7996 |
| SVM (rbf) | 0.7731 | 0.8304 | 0.8400 |
| Decision Tree (depth=8) | 0.7569 | 0.8110 | 0.7946 |

### Tuned models

| Model | Best Params | Accuracy | F1 | AUC |
|---|---|---|---|---|
| SVM (GridSearch) | rbf, C=1, gamma=1 | 0.7915 | 0.8456 | 0.8547 |
| **SVM (RandomSearch)** ⭐ | rbf, C=2.335, gamma=1.383 | **0.8015** | **0.8532** | **0.8594** |
| Decision Tree (GridSearch) | entropy, depth=10, leaf=50 | 0.7508 | 0.8076 | 0.8113 |

## Key Findings

- **Best model:** SVM (RBF kernel) tuned with **RandomSearch** — best on all three metrics. Its curved decision boundary fits the non-linear relationship between wine chemistry and quality.
- **SVM (RandomSearch) vs SVM (GridSearch):** Random Search found slightly better parameters (values *between* the grid points) and ran faster (12.5s vs 18.2s).
- **Overfitting trade-off:** the best (RandomSearch) SVM is more accurate but overfits (train 0.9888 / test 0.8015, gap 0.1873), while min_samples_leaf=50 made the Decision Tree generalize cleanly (gap 0.0191).
- **Most predictive feature:** `alcohol` — the Decision Tree's root split, confirming the Part 1 correlation analysis.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
