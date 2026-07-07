import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Dell\Downloads\Data.csv")

# Independent variables
x = dataset.iloc[:, :-1].values

# Dependent variable
y = dataset.iloc[:, 3].values

# Handle missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x , y, train_size=0.8,test_size=0.2,random_state=0)
