# NANO706 Homework 2 — Part 1

This repository contains a single Jupyter notebook, `Nano706_Homework2_part1.ipynb`, demonstrating core machine learning workflows on classic scikit-learn datasets and a small homework section on the Wine dataset.

## Contents

- `Nano706_Homework2_part1.ipynb`
- `iris_decision_tree` (Graphviz DOT file exported from a trained Decision Tree)
- `iris_decision_tree.png` (rendered decision tree image)

## What the notebook covers

- **Supervised learning**
  - Linear Regression on California Housing (1 feature for visualization)
  - Logistic Regression decision boundary on Iris (2 features)
  - Decision Tree on Iris with Graphviz export
  - Random Forest feature importances on Iris
  - SVM (linear kernel) decision boundaries on Iris (pairwise classes)
  - MLPClassifier on Iris with accuracy
- **Unsupervised learning**
  - K-Means clustering on Iris with k = 3, 4, 5
- **Homework (Wine dataset)**
  - Visualization of `alcohol` vs `malic_acid`
  - Standardization with `StandardScaler`
  - Train/test split
  - SVM baseline and accuracy
  - Decision Tree (with confusion matrix, classification report, feature importances, optional tree plot)
  - Logistic Regression with pipeline + GridSearchCV (confusion matrix, classification report, coefficient importances)

## Prerequisites

- Python 3.9+ (tested with 3.10)
- Jupyter
- Packages:
  - numpy, pandas, matplotlib, seaborn
  - scikit-learn
  - graphviz, pydot

Install Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn graphviz pydot
```

To render the decision tree image, you must also install the Graphviz system binaries:

- macOS (Homebrew):

```bash
brew install graphviz
```

- Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y graphviz
```

- Windows:

1) Download and install Graphviz from https://graphviz.org/download/
2) Add the Graphviz `bin` directory to your PATH (e.g., `C:\Program Files\Graphviz\bin`).

## How to run

1. Start Jupyter and open the notebook:

```bash
jupyter notebook Nano706_Homework2_part1.ipynb
```

2. Run cells in order. Plots will display inline. If Graphviz is installed, running the Decision Tree section will create/overwrite:

- `iris_decision_tree` (DOT file)
- `iris_decision_tree.png`

## Expected outputs

- Multiple inline figures for regression/classification/clustering sections.
- Console metrics such as accuracies and classification reports (e.g., SVM ≈ 0.98 on Wine, Decision Tree ≈ 0.96, Logistic Regression up to 1.00 depending on split/seed).
- `iris_decision_tree.png` in the repository root when Graphviz is available.

## Notes and troubleshooting

- If Graphviz rendering is skipped or you see errors like `ExecutableNotFound: failed to execute dot`, ensure Graphviz is installed and on your PATH (see Prerequisites).
- scikit-learn may emit a FutureWarning about `multi_class` in LogisticRegression; the default behavior will switch to multinomial in newer versions. You can safely omit the `multi_class` argument.
- The California Housing loader is `fetch_california_housing` (requires internet on first run to cache data). If you have network restrictions, consider downloading the dataset separately.

## Reproducibility tips

- The notebook uses `random_state=42` for train/test splits and some models to improve reproducibility.
- Exact metrics can vary slightly with library versions and platform.
