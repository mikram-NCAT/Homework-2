# NANO706 Homework 2 — Part 1

This repository contains a single Jupyter notebook, `Nano706_Homework2_part1.ipynb`, demonstrating core machine learning workflows on classic scikit-learn datasets and a small homework section on the Wine dataset.

## Contents

- `Nano706_Homework2_part1.ipynb
- classification report
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


##  outputs

- Multiple inline figures for regression/classification/clustering sections.
- <img width="646" height="511" alt="Regression" src="https://github.com/user-attachments/assets/2e18aa16-1b95-4b68-ae31-9e81f4d62c09" />
- Console metrics such as accuracies and classification reports
- (e.g., SVM ≈ 0.98 on Wine, Decision Tree ≈ 0.96, Logistic Regression up to 1.00 depending on split/seed).
- <img width="526" height="232" alt="classification report" src="https://github.com/user-attachments/assets/d2c2b0eb-339b-43a1-b15d-a483b6565ddd" />
- `iris_decision_tree.png` in the repository root
- <img width="1033" height="1053" alt="image" src="https://github.com/user-attachments/assets/e8a61236-5b87-402a-9ea2-46c6cdbc5cd2" />


## Notes and troubleshooting

- If Graphviz rendering is skipped or you see errors like `ExecutableNotFound: failed to execute dot`, ensure Graphviz is installed and on your PATH (see Prerequisites).
- scikit-learn may emit a FutureWarning about `multi_class` in LogisticRegression; the default behavior will switch to multinomial in newer versions. You can safely omit the `multi_class` argument.
- The California Housing loader is `fetch_california_housing` (requires internet on first run to cache data). If you have network restrictions, consider downloading the dataset separately.

## Reproducibility tips

- The notebook uses `random_state=42` for train/test splits and some models to improve reproducibility.
- Exact metrics can vary slightly with library versions and platform.

## Key Takeaways

Demonstrates the complete machine learning workflow: data preprocessing → model training → evaluation → visualization.

SVM and Logistic Regression performed best on the Wine dataset, with accuracies near or equal to 100%.

Graphviz decision tree rendering and reproducibility tips ensure consistent results across runs.
