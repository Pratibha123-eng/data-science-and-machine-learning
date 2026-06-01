# Bangalore House Price Prediction using Linear Regression

## Project Overview

This project predicts house prices in Bangalore using Machine Learning techniques. The primary objective is to build a predictive model that estimates property prices based on housing features using Linear Regression and Polynomial Regression.

The project demonstrates the complete machine learning workflow, including data loading, preprocessing, model training, evaluation, and performance comparison.

---

## Objectives

* Predict Bangalore house prices using historical housing data.
* Implement and evaluate a Linear Regression model.
* Apply feature scaling for data standardization.
* Measure model performance using standard regression metrics.
* Compare Linear Regression with Polynomial Regression.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Google Colab

---

## Dataset

The dataset is loaded from a Google Drive CSV file and contains housing-related features along with the target variable:

**Target Variable**

* `price`

**Input Features**

* Multiple housing attributes used for price prediction.

Dataset Source:

```python
path = "https://drive.google.com/uc?export=download&id=1xxDtrZKfuWQfl-6KA9XEd_eatitNPnkB"
```

---

## Project Workflow

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
```

Additional libraries used:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
```

---

### 2. Load Dataset

```python
df = pd.read_csv(path)
```

The dataset is loaded into a Pandas DataFrame and inspected before model development.

---

### 3. Feature Selection

Separate independent and dependent variables:

```python
x = df.drop('price', axis=1)
y = df['price']
```

---

### 4. Train-Test Split

The dataset is divided into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=51
)
```

* Training Data: 80%
* Testing Data: 20%

---

### 5. Feature Scaling

Standardization is performed using StandardScaler:

```python
sc = StandardScaler()
sc.fit(x_train)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
```

Feature scaling helps improve model stability and convergence.

---

### 6. Linear Regression Model

Train the Linear Regression model:

```python
lr = LinearRegression()
lr.fit(x_train, y_train)
```

Generate predictions:

```python
y_pred = lr.predict(x_test)
```

---

## Model Evaluation

### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

Measures average squared prediction error.

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

Provides prediction error in the same unit as house prices.

### R² Score

```python
r2_score(y_test, y_pred)
```

Measures the proportion of variance explained by the model.

### Model Accuracy

```python
lr.score(x_test, y_test)
```

Returns the coefficient of determination (R²).

---

## Polynomial Regression

To capture non-linear relationships, Polynomial Features are generated:

```python
poly_reg = PolynomialFeatures(degree=2)

x_train_poly = poly_reg.fit_transform(x_train)
x_test_poly = poly_reg.transform(x_test)
```

Train the model:

```python
lr.fit(x_train_poly, y_train)
```

Evaluate performance:

```python
lr.score(x_test_poly, y_test)
```

---

## Results

The notebook evaluates:

* Linear Regression Performance
* Polynomial Regression Performance
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

Polynomial Regression may improve prediction accuracy by modeling feature interactions and non-linear patterns.

---

## Project Structure

```
Bangalore-House-Price-Prediction/
│
├── Linear_Regression.ipynb
├── README.md
└── dataset.csv
```

---

## Machine Learning Concepts Demonstrated

* Supervised Learning
* Regression Analysis
* Feature Scaling
* Train-Test Split
* Model Evaluation
* Linear Regression
* Polynomial Regression
* Error Metrics

---

## Future Improvements

* Hyperparameter tuning
* Feature engineering
* Cross-validation
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* XGBoost Regressor
* Model deployment using Flask or Streamlit

---

## Conclusion

This project demonstrates a practical implementation of house price prediction using Linear Regression and Polynomial Regression. The workflow covers data preprocessing, model training, prediction, and evaluation, providing a solid foundation for real-estate price forecasting applications and further machine learning experimentation.

---

## Author

Machine Learning Regression Project

Built using Python, Scikit-learn, and Google Colab.
