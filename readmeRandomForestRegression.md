# 🌲 Bangalore House Price Prediction using Random Forest Regression

## Project Overview

This project predicts house prices in Bangalore using the **Random Forest Regression** algorithm. Random Forest is an ensemble learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

The notebook demonstrates the complete machine learning workflow, including data preprocessing, model training, prediction, and performance evaluation.

---

## 🎯 Objectives

* Predict Bangalore house prices using housing-related features.
* Implement Random Forest Regression for price prediction.
* Evaluate model performance using regression metrics.
* Compare Random Forest performance with traditional regression techniques.
* Understand ensemble learning for regression problems.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Google Colab

---

## 📊 Dataset

The dataset contains Bangalore housing information with multiple property-related features.

### Target Variable

* `price`

### Features

* Housing attributes used to estimate property prices.
* Numerical and categorical features related to real estate properties.

---

## 🔄 Project Workflow

### 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
```

Additional libraries:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

---

### 2. Load Dataset

```python
df = pd.read_csv("dataset.csv")
```

Inspect and explore the dataset before model training.

---

### 3. Feature Selection

Separate independent and dependent variables:

```python
X = df.drop("price", axis=1)
y = df["price"]
```

---

### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* Training Data: 80%
* Testing Data: 20%

---

### 5. Train Random Forest Regressor

```python
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
```

---

### 6. Make Predictions

```python
y_pred = rf.predict(X_test)
```

---

## 📈 Model Evaluation

### R² Score

```python
r2_score(y_test, y_pred)
```

Measures how well the model explains the variance in house prices.

### Mean Squared Error (MSE)

```python
mean_squared_error(y_test, y_pred)
```

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

Provides prediction error in the same unit as house prices.

---

## 🌲 Why Random Forest Regression?

Random Forest Regression offers several advantages:

* Handles non-linear relationships effectively.
* Reduces overfitting through ensemble learning.
* Works well with large datasets.
* Provides feature importance scores.
* Robust against noise and outliers.

---

## 📊 Feature Importance

Random Forest can identify the most influential features affecting house prices:

```python
feature_importance = rf.feature_importances_
```

This helps understand which housing attributes contribute most to price prediction.

---

## 📁 Project Structure

```text
Bangalore-House-Price-Prediction-RandomForest/
│
├── Random_forest_regression.ipynb
├── README.md
├── dataset.csv
└── requirements.txt
```

---

## 🧠 Machine Learning Concepts Covered

* Supervised Learning
* Regression Analysis
* Ensemble Learning
* Decision Trees
* Random Forest Regression
* Feature Importance
* Model Evaluation
* Train-Test Split

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Feature engineering
* Outlier detection and handling
* Comparison with:

  * Linear Regression
  * Polynomial Regression
  * Support Vector Regression (SVR)
  * XGBoost Regressor
* Model deployment using Streamlit or Flask

---

## 📌 Results

The model performance is evaluated using:

* R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

These metrics provide insights into prediction accuracy and model effectiveness.

---

## 🏆 Conclusion

This project demonstrates the application of Random Forest Regression for Bangalore house price prediction. By leveraging multiple decision trees, the model captures complex relationships within housing data and delivers robust predictions.

Random Forest Regression is a powerful approach for real-world price forecasting problems due to its high accuracy, flexibility, and resistance to overfitting.

---

## 👨‍💻 Author

**Your Name**

Machine Learning Project built with Python, Scikit-learn, and Google Colab.
