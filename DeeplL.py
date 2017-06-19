#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:10:47 2017

@author: ryad
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
labelencoder_X_Gender = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onhotencoder = OneHotEncoder(categorical_features= [1])
X = onhotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Builiding the networks 

import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))#input layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))#hidden layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))#output layer


classifier.compile(optimizer = 'nadam',loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam for stochastic gradient discent , optimizing the entropye , accuracy metric

classifier.fit(X_train,y_train , batch_size = 10 , epochs = 20)

y_pred = classifier.predict(X_test)
cm_y = y_pred
for i in range(0,2000):
    if (y_pred[i]>0.5):
        cm_y[i] = 1
    else :
       cm_y[i]=0

#Making the consfusion matrix
from sklearn.metrics import confusion_matrix
cm_pred = confusion_matrix(y_test,cm_y)
print((cm_pred[0,0]+cm_pred[1,1])/2000)

