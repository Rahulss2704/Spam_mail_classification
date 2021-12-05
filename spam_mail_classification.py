#Spam mail Classification using SVM

#Import Libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv("spam.csv")
dataset.where((pd.notnull(dataset)),"")
dataset.head()
dataset.loc[dataset["Category"]=="spam","Category",]=1
dataset.loc[dataset["Category"]=="ham","Category",]=0
X = dataset["Message"]
y = dataset["Category"]

#splitting dataset into the Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)

#feature extraction
#Transform text data to feature vectors that can be used as input to the svm model using TfidfVectoriser
from sklearn.feature_extraction.text import TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1,stop_words=("english"),lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert y_train & y_test values as integers
y_train = y_train.astype("int")
y_test = y_test.astype("int")

#Training the SVM model 
from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train_features,y_train)

#prediction on training data
from sklearn.metrics import accuracy_score
prediction_training_data = classifier.predict(X_train_features)
accuracy_training_data = accuracy_score(y_train,prediction_training_data)
print("Accuracy training data: ",accuracy_training_data)

prediction_test_data = classifier.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test,prediction_test_data)
print("Accuracy test data: ",accuracy_test_data)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_test_data)
print(cm)
