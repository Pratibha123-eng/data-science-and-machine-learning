# 🏠 Bangalore House Price Prediction using K-Nearest Neighbors (KNN) Regression

## Project Overview

This project predicts house prices in Bangalore using the **K-Nearest Neighbors (KNN) Regression** algorithm. KNN Regression is a non-parametric machine learning technique that predicts values based on the average of the nearest neighboring data points.

The notebook demonstrates data preprocessing, feature scaling, model training, prediction, and performance evaluation for real estate price prediction.

---

## 🎯 Objectives

* Predict Bangalore house prices using KNN Regression.
* Understand instance-based learning techniques.
* Evaluate model performance using regression metrics.
* Analyze the impact of different K values on prediction accuracy.
* Compare KNN Regression with other regression algorithms.

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

The dataset contains Bangalore housing information and property-related features used for predicting house prices.

### Target Variable

* `price`

### Features

* Housing attributes such as area, rooms, location-based factors, and other property characteristics.

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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
```

---

### 2. Load Dataset

```python
df = pd.read_csv("dataset.csv")
```

The dataset is loaded and inspected before model development.

---

### 3. Feature Selection

```python
X = df.drop("price", axis=1)
y = df["price"]
```

Separate input features and target variable.

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

### 5. Feature Scaling

Since KNN relies on distance calculations, feature scaling is essential.

```python
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
```

---

### 6. Train KNN Regressor

```python
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_sc, y_train)
```

---

### 7. Make Predictions

```python
y_pred = knn.predict(X_test_sc)
```

---

## 📈 Model Evaluation

### R² Score

```python
r2_score(y_test, y_pred)
```

Measures how well the model explains variance in house prices.

### Mean Squared Error (MSE)

```python
mean_squared_error(y_test, y_pred)
```

Measures average prediction error.

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

Provides prediction error in the same unit as house prices.

---

## 🔍 Understanding KNN Regression

KNN Regression predicts a value by:

1. Finding the K nearest data points.
2. Calculating the average of their target values.
3. Returning the averaged value as the prediction.

### Example

If:

```text
K = 5
```

The model considers the 5 closest houses and predicts the price based on their average price.

---

## 📊 Choosing the Best K Value

Different values of K can significantly impact model performance.

Example:

```python
for k in range(1, 21):
    model = KNeighborsRegressor(n_neighbors=k)
```

Smaller K values:

* More sensitive to noise.
* Higher variance.

Larger K values:

* More stable predictions.
* Higher bias.

---

## 📁 Project Structure

```text
Bangalore-House-Price-Prediction-KNN/
│
├── K_nearest_neighbour_regression.ipynb
├── README.md
├── dataset.csv
└── requirements.txt
```

---

## 🧠 Machine Learning Concepts Covered

* Supervised Learning
* Regression Analysis
* K-Nearest Neighbors (KNN)
* Distance-Based Learning
* Feature Scaling
* Model Evaluation
* Hyperparameter Tuning

---

## 🚀 Future Improvements

* Optimize K using GridSearchCV
* Weighted KNN Regression
* Feature Engineering
* Outlier Detection
* Cross Validation
* Compare with:

  * Linear Regression
  * Random Forest Regression
  * Support Vector Regression (SVR)
  * XGBoost Regression
* Deploy using Streamlit

---

## 📌 Results

The model is evaluated using:

* R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

These metrics help determine how accurately the model predicts house prices.

---

## 🏆 Conclusion

This project demonstrates the implementation of K-Nearest Neighbors Regression for Bangalore house price prediction. By leveraging similarity-based learning and feature scaling, the model can estimate property prices based on neighboring housing records.

KNN Regression provides a simple yet effective approach for regression problems where similar observations tend to have similar target values.

---

## 👨‍💻 Author

**Your Name**

Machine Learning Project built with Python, Scikit-learn, and Google Colab.
