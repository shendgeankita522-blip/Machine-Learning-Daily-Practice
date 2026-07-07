# Importing the libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# Importing the dataset 
Dataset = pd.read_csv(r"C:\Users\Dell\Downloads\Future prediction1.csv")
X = Dataset.iloc[:, [2,3]].values
y = Dataset.iloc[:, -1].values

# Splitting the  dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_test,y_train = 


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#This is to get the Model accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_true, y_pred)
print(ac)

bias = classifiesr.score(X_train,y_train)
print(bias)

