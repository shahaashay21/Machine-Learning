# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:21:15 2020

@author: aashays
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encoding the Independent variable
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train).astype(float)
X_test = sc_X.transform(X_test).astype(float)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1).astype(float))"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test);