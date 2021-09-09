#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 00:33:26 2019

@author: ilyasse
"""

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

data=pd.read_csv('clean_data_test.csv')
data=np.matrix(data,float)
X=data[:,1:13];
Y=data[:,13:17];
#
#a=outliers_iqr(X)
#
#a[0,:] = list(dict.fromkeys(a[0,:]))
#print(a) 
#y=np.where(outlier[:,:]==True)
#X = np.delete(X, y[0], axis=0)
#Y = np.delete(Y, (y[0]), axis=0)
#
#X_train=X[:-12,:]
#X_test=X[-12:,:]
#
#Y_train=Y[:-12,:]
#Y_test=Y[-12:,:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)


# #############################################################################


nn_reg=MLPRegressor(hidden_layer_sizes=(128,) ,activation='relu',solver='sgd',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.003,alpha=0.35)


nn_reg.fit(X_train, Y_train)
Y_pred=nn_reg.predict(X_test)

x=[i for i in range(0,12)]
plt.plot(x,Y_test[0:12,0],'b',Y_pred[0:12,0],'r')
plt.title('Energie')
plt.grid()
#x=np.matrix(x)
#a=np.where(Y_pred==241.16)

mean_absolute_error(Y_test[:,0], Y_pred[:,0])
#np.sum(np.multiply((Y_test-Y_pred),(Y_test-Y_pred)))
#accuracy_score(Y_test[:,0], Y_pred[:,0])
r2_score(Y_test[:,0], Y_pred[:,0])
mean_squared_error(Y_test[:,0], Y_pred[:,0])


plt.figure(2)
x=[i for i in range(0,12)]
plt.plot(x,Y_test[0:12,1],'b',Y_pred[0:12,1],'r')
plt.title('Pression')
plt.grid()
#x=np.matrix(x)
#a=np.where(Y_pred==241.16)

#mean_absolute_error(Y_test[:,2], Y_pred[:,2])
##np.sum(np.multiply((Y_test-Y_pred),(Y_test-Y_pred)))
##accuracy_score(Y_test[:,0], Y_pred[:,0])
#r2_score(Y_test[:,1], Y_pred[:,1])
#mean_squared_error(Y_test[:,1], Y_pred[:,1])


plt.figure(3)

x=[i for i in range(0,12)]
plt.plot(x,Y_test[0:12,2],'b',Y_pred[0:12,2],'r')
plt.title('Amplitude')
plt.grid()



plt.figure(4)

x=[i for i in range(0,12)]
plt.plot(x,Y_test[0:12,3],'b',Y_pred[0:12,3],'r')
plt.grid()
plt.title('Width (mm)')

#x=np.matrix(x)
#a=np.where(Y_pred==241.16)

#mean_absolute_error(Y_test[:,2], Y_pred[:,2])
##np.sum(np.multiply((Y_test-Y_pred),(Y_test-Y_pred)))
##accuracy_score(Y_test[:,0], Y_pred[:,0])
#r2_score(Y_test[:,1], Y_pred[:,1])
#mean_squared_error(Y_test[:,1], Y_pred[:,1])

