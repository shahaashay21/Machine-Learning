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

# Building the optimal model using Backward Elimination (Method 1)
import statsmodels.formula.api as sm
# Equation is Yo = Bo + B1X1 + BnXn
# We don't have Bo in our equation, in order get it we need to assume there is BoXo and Xo is always 1
# For that, we need to prepend one column with value 1
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# Let's start BE process
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

# Same as Method 1 but in a function with a loop
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Define significance level
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


# Predicting the Test set results after backward elimination process
X_opt_train, X_opt_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)
regressor_with_be = LinearRegression()
regressor_with_be.fit(X_opt_train, y_train)
y_pred_with_be = regressor_with_be.predict(X_opt_test);


# Backward Elimination with p-values and Adjusted R Squared (Method 2)
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled_Adj_R = backwardElimination(X_opt, SL)

# Predicting the Test set results after Backward Elimination with p-values and Adjusted R Squared
X_opt_adj_r_train, X_opt_adj_r_test, y_train, y_test = train_test_split(X_Modeled_Adj_R, y, test_size = 0.2, random_state = 0)
regressor_with_be_adj_r = LinearRegression()
regressor_with_be_adj_r.fit(X_opt_adj_r_train, y_train)
y_pred_with_be_adj_r = regressor_with_be_adj_r.predict(X_opt_adj_r_test);