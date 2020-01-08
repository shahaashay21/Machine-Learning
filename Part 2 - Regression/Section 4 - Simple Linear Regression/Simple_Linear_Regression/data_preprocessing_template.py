# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 23:01:22 2020

@author: aashays
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#splitting the dataset and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train).astype(float)
X_test = sc_X.transform(X_test).astype(float)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1).astype(float))"""