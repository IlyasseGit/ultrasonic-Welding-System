#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:33:14 2019

@author: ilyasse
"""

print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
data = np.loadtxt('ex1data2.txt', delimiter=",");


# Use only one feature
X=data[:,0:2];
Y=data[:,2];
# Create linear regression object
regr = linear_model.LinearRegression()
X=X.reshape((47,2))
Y=Y.reshape((47,1))
# Train the model using the training sets
regr.fit(X, Y)

# Make predictions using the testing set
Y_pred = regr.predict(X)

# The coefficients
print('Coefficients:theta1:\n', regr.coef_,'theta0 :\n',regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y, Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y, Y_pred))
# Plot outputs
