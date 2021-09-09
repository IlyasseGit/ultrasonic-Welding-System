#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 03:03:51 2019

@author: ilyasse
"""

import numpy as np
import pandas as pd

#def test_int(T,i,j):
#    return T[i,j]-int(T[i,j])==0

data = pd.read_excel (r'DATA ULtrasonic welding Process.xlsx') 
fdf = pd.DataFrame(data, columns= ['Left Side','Right Side','Energy (J)','Pressure (bar)','Amplitude (%)','Width (mm)'])

#fdf.drop_duplicates(keep=False, inplace=True)
fdf.drop_duplicates(keep='last', inplace=True)

fdf.dropna(thresh=4, inplace=True)
fdf.fillna(0, inplace=True)

T=np.zeros((len(fdf),18),object);
final=np.matrix(fdf)

for i in range(len(fdf)):
    
    
    
    for j in range(2):
        
        a=str(final[i,j])
        if a.find('+')>-1:
            
            if a.find('*')==-1:
                y=a.split('+')
                z=''
                for t in range(a.count('+')+1):
                   z=z+ y[t]+'*1'+'+'
                
                final[i,j]=z[:len(z)-1]
            
            if a.count('*')>0:
                y=a.split('+')
                z=''
                for t in range(a.count('+')+1):
                    if y[t].count('*')==1:
                        z=z+y[t]+'+'
                    else:
                        z=z+y[t]+'*1'+'+'
                        
                final[i,j]=z[:len(z)-1]
                        
                
        else:
            if a.find('*')==-1:
                final[i,j]=a+'*1'




for i in range(len(fdf)):
    
    
    
    for j in range(2):
        a=final[i,j]
        if type(a)==str:
            a=a.replace("+", "*")
            a=a.replace(",", ".")
            y=a.split('*')
        
            for k in range(len(y)):
                if y[k].isdecimal()==True:
                    T[i,k+ 8*j]=int(y[k])
                else:
                    T[i,k+ 8*j]=float(y[k])


tab_f=np.zeros((len(fdf),16),object)

for i in range(len(fdf)):    
    j=0
    n=0
    m=0
    while(j<16):
        if j<8:            
            if (type(T[i,j])==float and type(T[i,j+1])==int):
                
                for k in range(T[i,j+1]):
                    
                    tab_f[i,k+n]=T[i,j]
                    
                n+=T[i,j+1]
                
                j=j+2
                continue
            if(type(T[i,j])==int and type(T[i,j+1])==float):
                for k in range(T[i,j]):
                    tab_f[i,k+n]=T[i,j+1]
                
                n+=T[i,j]
            else:
                if T[i,j]<=T[i,j+1]:
                    for k in range(int(T[i,j])):
                        tab_f[i,k+n]=T[i,j+1]
                    n+=int(T[i,j])
                else:
                    for k in range(int(T[i,j+1])):
                        tab_f[i,k+n]=T[i,j]
                    n+=int(T[i,j+1])
                
        else:
            
            if (type(T[i,j])==float and type(T[i,j+1])==int):
                
                for k in range(T[i,j+1]):
                    tab_f[i,6+k+m]=T[i,j]
                m+=int(T[i,j+1])
                j=j+2
                continue
            if(type(T[i,j])==int and type(T[i,j+1])==float):
                for k in range(T[i,j]):
                    tab_f[i,6+k+m]=T[i,j+1]
                
                m+=int(T[i,j])
            
            else:
                

                if T[i,j]<=T[i,j+1]:
                    for k in range(int(T[i,j])):
                        tab_f[i,6+k+m]=T[i,j+1]
                    m+=int(T[i,j])
                else:
                    print(j)
                    for k in range(int(T[i,j+1])):
                        tab_f[i,6+k+m]=T[i,j]
                    m+=int(T[i,j+1])
                    
                
        
        j=j+2

tab_f=np.matrix(tab_f,float)
M=np.matrix(final[:,2:6],float)
tab_f[:,12:16]=M

########################## Outlier ##############################
def find_outliers(df):
    df_zscore = (df - df.mean())/df.std()

    return df_zscore > 3
outlier=find_outliers(tab_f)

z=np.where(tab_f[:,:]==True)
tab_f = np.delete(tab_f, z[0], axis=0)

##################################################
df =pd.DataFrame(tab_f, columns = ["wire1","wire2","wire3","wire4","wire5","wire6","wire7","wire8","wire9","wire10","wire11","wire12",'Energy (J)','Pressure (bar)','Amplitude (%)','Width (mm)'])
df.drop_duplicates(['Energy (J)','Pressure (bar)','Amplitude (%)','Width (mm)'],keep='first', inplace=True)

df.to_csv('clean_data_test3.csv')

