# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:03:00 2020

@author: aashays

No Train and Test data because we are working with the assuption that new data has experience level between 6 and 7 and salary is 160,000
Now here we are checking whether the above 6.5 level with 160,000 is right or not
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Split the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train).astype(float)
X_test = sc_X.transform(X_test).astype(float)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1).astype(float))"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visulising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Visulising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
linear_model_predition = lin_reg.predict(np.array(6.5).reshape(-1, 1))

# Predicting a new result with Polynomial Regression
polynomial_model_predition = lin_reg2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1,1)))