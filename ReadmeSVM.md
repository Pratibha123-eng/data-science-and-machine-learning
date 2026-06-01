# 🎗️ Breast Cancer Classification using Support Vector Machine (SVM)

## Project Overview

This project demonstrates the use of a Support Vector Machine (SVM) classifier to predict whether a breast cancer tumor is malignant or benign using the Breast Cancer Wisconsin dataset from Scikit-learn.

The notebook covers the complete machine learning workflow, including data loading, preprocessing, train-test splitting, feature scaling, model training, and performance evaluation.

---

## Objectives

* Build a binary classification model using Support Vector Machine (SVM).
* Predict breast cancer diagnosis based on tumor characteristics.
* Compare model performance before and after feature scaling.
* Evaluate classification accuracy on unseen test data.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Google Colab / Jupyter Notebook

---

## Dataset

The project uses the built-in Breast Cancer Wisconsin Dataset available in Scikit-learn.

### Dataset Information

* Number of Samples: 569
* Number of Features: 30
* Classes:

  * Malignant (0)
  * Benign (1)

### Load Dataset

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

---

## Project Workflow

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
```

### 2. Load Dataset

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

### 3. Create DataFrame

```python
df = pd.DataFrame(
    np.c_[data.data, data.target],
    columns=list(data.feature_names) + ['target']
)
```

### 4. Feature Selection

```python
X = df.drop(columns=['target'])
y = df['target']
```

### 5. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2020
)
```

### 6. Train SVM Classifier

```python
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
```

---

## Model Training

### Without Feature Scaling

```python
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
```

Evaluate accuracy:

```python
classifier.score(X_test, y_test)
```

---

## Feature Scaling

Since SVM is sensitive to feature magnitude, standardization is applied.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

---

## SVM with Scaled Features

```python
classifier.fit(X_train_sc, y_train)
classifier.score(X_test_sc, y_test)
```

Feature scaling generally improves SVM performance by ensuring all features contribute equally to the decision boundary.

---

## Performance Evaluation

The project evaluates the classifier using:

* Accuracy Score
* Training Accuracy
* Testing Accuracy

```python
from sklearn.metrics import accuracy_score
```

---

## Project Structure

```text
Breast-Cancer-SVM/
│
├── support_vector_machine.ipynb
├── README.md
└── requirements.txt
```

---

## Machine Learning Concepts Covered

* Supervised Learning
* Binary Classification
* Support Vector Machine (SVM)
* Radial Basis Function (RBF) Kernel
* Feature Scaling
* Data Preprocessing
* Train-Test Split
* Model Evaluation

---

## Why Use SVM?

Support Vector Machines are effective for classification tasks because they:

* Work well with high-dimensional datasets.
* Create optimal decision boundaries.
* Handle non-linear classification using kernels.
* Provide strong performance on medical datasets.

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Compare different kernels:

  * Linear
  * Polynomial
  * RBF
  * Sigmoid
* Confusion Matrix Visualization
* ROC Curve Analysis
* Precision, Recall, and F1 Score Evaluation

---

## Results

The notebook compares classification performance:

✅ Without Feature Scaling

✅ With Feature Scaling

The scaled SVM model typically achieves better classification performance because SVM relies on distance-based calculations.

---

## Conclusion

This project demonstrates how Support Vector Machines can be used for breast cancer diagnosis. By applying feature scaling and the RBF kernel, the model effectively classifies tumors as malignant or benign, showcasing the power of SVMs in medical prediction tasks.

---

## Author

Machine Learning Classification Project

Built using Python, Scikit-learn, and Google Colab.
