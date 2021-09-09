#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:15:21 2019

@author: ilyasse
"""

print(__doc__)
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

def find_outliers(df):
    df_zscore = (df - df.mean())/df.std()

    return df_zscore > 3

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
# Fit regression model
svr_rbf =MultiOutputRegressor( SVR(kernel='rbf', C=100, gamma=0.025, epsilon=.2))
#svr_lin = SVR(kernel='linear', C=100, gamma='auto')
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#               coef0=1)

svr_rbf.fit(X_train, Y_train)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)
Y_pred = svr_rbf.predict(X_test)

# #############################################################################
# Look at the results
# Plot Energie

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
