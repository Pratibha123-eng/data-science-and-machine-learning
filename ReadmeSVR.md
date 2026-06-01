# 🏠 Bangalore House Price Prediction using Support Vector Regression (SVR)

## Project Overview

This project predicts house prices in Bangalore using **Support Vector Regression (SVR)**, a powerful machine learning algorithm capable of modeling both linear and non-linear relationships.

The notebook demonstrates the complete machine learning workflow, including data preprocessing, feature scaling, model training, performance evaluation, and comparison of different SVR kernels.

---

## 🎯 Objectives

* Predict Bangalore house prices using housing features.
* Implement Support Vector Regression (SVR).
* Compare multiple SVR kernels:

  * RBF Kernel
  * Linear Kernel
  * Polynomial Kernel
* Evaluate model performance using R² Score and RMSE.
* Understand the impact of feature scaling on SVR performance.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Google Colab

---

## 📊 Dataset

The dataset contains Bangalore housing information with various features affecting property prices.

### Target Variable

* `price`

### Features

* Multiple housing-related attributes used for price prediction.

Dataset Source:

```python
path = "https://drive.google.com/uc?export=download&id=1xxDtrZKfuWQfl-6KA9XEd_eatitNPnkB"
```

---

## 🔄 Project Workflow

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
```

### 2. Load Dataset

```python
df = pd.read_csv(path)
```

### 3. Feature Selection

```python
X = df.drop('price', axis=1)
y = df['price']
```

### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=51
)
```

### 5. Feature Scaling

Since SVR is highly sensitive to feature magnitudes, standardization is applied.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

---

## 🤖 Support Vector Regression Models

### SVR with RBF Kernel

```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train_sc, y_train)

y_pred = svr.predict(X_test_sc)
```

### SVR with Linear Kernel

```python
svr = SVR(kernel='linear')
svr.fit(X_train_sc, y_train)

y_pred = svr.predict(X_test_sc)
```

### SVR with Polynomial Kernel

```python
svr = SVR(kernel='poly', degree=2)
svr.fit(X_train_sc, y_train)

y_pred = svr.predict(X_test_sc)
```

---

## 📈 Model Evaluation

### R² Score

Measures how well the model explains the variance in house prices.

```python
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
```

### Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

### Root Mean Squared Error (RMSE)

```python
import math

rmse = math.sqrt(mse)
```

RMSE provides prediction error in the same units as house prices.

---

## 📁 Project Structure

```text
Bangalore-House-Price-Prediction-SVR/
│
├── bangolre_house_prediction_SVR.ipynb
├── README.md
└── dataset.csv
```

---

## 🧠 Machine Learning Concepts Covered

* Supervised Learning
* Regression Analysis
* Support Vector Regression (SVR)
* Kernel Methods
* Feature Scaling
* Data Preprocessing
* Train-Test Split
* Model Evaluation

---

## 🔍 Kernel Comparison

The project compares three different SVR kernels:

| Kernel     | Purpose                                       |
| ---------- | --------------------------------------------- |
| RBF        | Captures complex non-linear relationships     |
| Linear     | Suitable for linear relationships             |
| Polynomial | Models curved relationships between variables |

This comparison helps identify the best-performing kernel for housing price prediction.

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Feature engineering
* Outlier detection and removal
* Comparison with:

  * Linear Regression
  * Random Forest Regressor
  * XGBoost Regressor
* Streamlit Web Application Deployment

---

## 📌 Results

The notebook evaluates:

* R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

for multiple SVR kernels to determine the most effective model for Bangalore house price prediction.

---

## ✅ Conclusion

This project demonstrates the use of Support Vector Regression for real estate price prediction. By applying feature scaling and experimenting with different kernels, the model captures relationships between housing features and property prices, providing valuable insights into machine learning-based price forecasting.

---

## 👨‍💻 Author

**Your Name**

Machine Learning Project built with Python, Scikit-learn, and Google Colab.
