# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:57:56 2020

@author: aashays
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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


# Fitting the regression model to the dataset
# CREATE YOUR REGRESSOR HERE



# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(np.array(6.5).reshape(-1,1))

# Visulising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = "blue")
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visulising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()