# XGBoost – Classification & Regression

## Overview

This repository contains Jupyter Notebooks demonstrating **XGBoost (Extreme Gradient Boosting)** for both **classification** and **regression** tasks. XGBoost is an optimized boosting algorithm known for its speed, regularization, and strong performance on structured (tabular) data.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. XGBoost Classification  
4. XGBoost Regression  
5. Model Evaluation   

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## Project Structure

- `XgboostBoost_Classification.ipynb` – XGBoost applied to a classification problem  
- `Xgboost_Regression.ipynb` – XGBoost applied to a regression problem  

---

## XGBoost Classification

### `XgboostBoost_Classification.ipynb`

This notebook demonstrates **XGBoost Classifier** for classification tasks.

Key points:
- Uses gradient boosting with regularization
- Handles non-linear relationships effectively
- Works well with imbalanced and complex datasets

Basic commands used:
```python
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
```

---

## XGBoost Regression

### `Xgboost_Regression.ipynb`

This notebook applies **XGBoost Regressor** for predicting continuous target values.

Key points:
- Sequentially builds trees to reduce residual errors
- Includes regularization to prevent overfitting
- Performs well on structured datasets

Basic commands used:
```python
from xgboost import XGBRegressor

xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
```

---

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

