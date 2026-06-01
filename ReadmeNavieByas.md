# 📧 Email Spam Detection using Naive Bayes Classifier

## Project Overview

This project implements a **Naive Bayes Classifier** to detect whether an email/message is **Spam** or **Ham (Not Spam)**. Naive Bayes is a probabilistic machine learning algorithm widely used in text classification tasks due to its simplicity, efficiency, and strong performance on large datasets.

The notebook demonstrates data preprocessing, text vectorization, model training, prediction, and evaluation.

---

## 🎯 Objectives

* Build a spam detection system using Naive Bayes Classification.
* Classify messages as Spam or Ham.
* Convert text data into numerical features.
* Evaluate model performance using classification metrics.
* Understand probabilistic machine learning techniques.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Google Colab / Jupyter Notebook

---

## 📊 Dataset

The dataset contains labeled text messages.

### Features

* Message/Text Content

### Target Variable

* Spam (1)
* Ham / Not Spam (0)

Example:

| Message                | Label |
| ---------------------- | ----- |
| Win a free iPhone now! | Spam  |
| Meeting at 5 PM today  | Ham   |

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
```

---

### 2. Load Dataset

```python
df = pd.read_csv("spam.csv")
```

Inspect and clean the dataset before model training.

---

### 3. Data Preprocessing

Convert text messages into numerical vectors.

```python
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])
y = df["label"]
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

### 5. Train Naive Bayes Model

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)
```

---

### 6. Make Predictions

```python
y_pred = model.predict(X_test)
```

---

## 📈 Model Evaluation

### Accuracy Score

```python
accuracy_score(y_test, y_pred)
```

Measures the percentage of correctly classified messages.

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
```

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Metrics include:

* Precision
* Recall
* F1-Score
* Support

---

## 🧠 Understanding Naive Bayes

Naive Bayes uses **Bayes' Theorem** and assumes that features are independent of each other.

P(A|B)=\frac{P(B|A)P(A)}{P(B)

The classifier calculates the probability of a message belonging to a class and predicts the class with the highest probability.

---

## 📁 Project Structure

```text
Naive-Bayes-Spam-Detection/
│
├── Naive_Bayas_classifier.ipynb
├── README.md
├── spam.csv
└── requirements.txt
```

---

## 📚 Machine Learning Concepts Covered

* Supervised Learning
* Text Classification
* Natural Language Processing (NLP)
* Naive Bayes Classification
* Feature Extraction
* Count Vectorization
* Probability-Based Learning
* Model Evaluation

---

## ✅ Advantages of Naive Bayes

* Fast and efficient
* Works well with text data
* Requires less training data
* Easy to implement
* Performs well on spam detection tasks

---

## ⚠️ Limitations

* Assumes feature independence
* May struggle with highly correlated features
* Sensitive to data quality

---

## 🚀 Future Improvements

* TF-IDF Vectorization
* Hyperparameter Tuning
* Cross Validation
* Email Subject Analysis
* Deep Learning Approaches
* Deployment using Flask or Streamlit
* Real-Time Spam Detection Application

---

## 📌 Results

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

These metrics help assess the effectiveness of spam detection.

---

## 🏆 Conclusion

This project demonstrates the use of the Naive Bayes algorithm for email and message spam detection. By converting text into numerical features and applying probabilistic classification, the model can efficiently distinguish between spam and legitimate messages.

Naive Bayes remains one of the most effective baseline algorithms for text classification problems.

---

## 👨‍💻 Author

**Your Name**

Machine Learning Project built with Python, Scikit-learn, and Google Colab.
