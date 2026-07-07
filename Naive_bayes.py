

# Naives Bayes

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
Dataset = pd.read_csv(r"D:\Daily_Class\june\1st - Navie Bayes\1st - Navie Bayes\project\adult.csv")

X = Dataset.iloc[:, [2,3]].values
y = Dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

X[:, 0] = le1.fit_transform(X[:, 0])
X[:, 1] = le2.fit_transform(X[:, 1])

X = X.astype(float)

# Splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# Feature Scaling
from sklearn.preprocessing import Normalizer

sc = Normalizer()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy Score
from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)
print("Accuracy =", ac)

# Bias
bias = classifier.score(X_train, y_train)
print("Bias =", bias)