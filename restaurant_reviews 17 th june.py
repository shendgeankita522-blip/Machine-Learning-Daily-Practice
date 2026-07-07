# Natural Language Processing 

# Importing the libraries 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# Importing the dataset

dataset = pd.read_csv(r"D:\Daily_Class\june\Restaurant_Reviews.tsv",delimiter = '\t' , quoting = 0)

# Cleaning the texts 
import re 
import nltk 

# nltk.download ('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review
              if word not in set(stopwords.words('english'))]

    review = ' '.join(review)
    corpus.append(review)

# creating the bag of Words model l

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values


# Splitting the dataset  into Training set and Test ste 
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.20 , random_state = 0)

from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier()
classifier.fit(X_train , y_train)


y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test , y_pred)
print(cm)




