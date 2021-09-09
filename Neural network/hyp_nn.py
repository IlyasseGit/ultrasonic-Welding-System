#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:50:27 2019

@author: ilyasse
"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

from sklearn.svm import SVR
import numpy as np
data=pd.read_csv('clean_data_test.csv')
data=np.matrix(data,float)
X=data[:,1:13];
Y=data[:,13:17];


parameters = {'estimator__hidden_layer_sizes':[(100,),(120,)],'estimator__activation': ('identity', 'logistic', 'tanh', 'relu'), 'estimator__learning_rate_init': [1e-7, 1e-4,0.003,0.01,0.025],'estimator__alpha':[0.1,0.2,0.35,0.4,0.5]}
nn_reg=MLPRegressor()

clf = GridSearchCV( nn_reg, param_grid=parameters)
clf.fit(X,Y)
clf.best_params_

