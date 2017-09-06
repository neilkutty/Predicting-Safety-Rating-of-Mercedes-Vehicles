

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:13:51 2017

@author: Neil Kutty 
"""

#%%
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


sb.set_style("darkgrid")

merc = pd.read_csv('train.csv')


#%%
def convert_data(data):
    num = data.select_dtypes(include = ['float64', 'int64']) 
    non = data.select_dtypes(exclude = ['float64', 'int64'])
    j = pd.DataFrame()
    k = pd.DataFrame()
    for column in non:
        j[column] = pd.Categorical(non[column])
        k[column] = j[column].cat.codes
    cbind = pd.concat([num,k], axis=1)
    return cbind


nmerc = convert_data(merc)    


#%%
plt.rcParams['figure.figsize']=(10,9)
corrdf = merc.corr()
ycorr = corrdf[['y']]
ycorr = ycorr[ycorr.y.abs() >= 0.2]
ycorr = ycorr.sort_values(by='y')

ax = ycorr.drop(['y']).plot(kind='barh')
ax.set_ylabel('Variable')
ax.set_xlabel('Correlation to Outcome')



#%%
plt.rcParams['figure.figsize']=(10,9)
corrdf = merc.corr()
ycorr = corrdf[['y']]
ycorr = ycorr[ycorr.y.abs() > 0.3]
ycorr = ycorr.sort_values(by='y')
ax = ycorr.drop(['y']).plot(kind='barh')
ax.set_ylabel('Variable')
ax.set_xlabel('Correlation to Outcome')


#%% Prep Data for Modeling
# Create train and test sets

traindf, testdf = train_test_split(nmerc, test_size=0.3)

X = traindf.drop('y',axis=1).drop('ID',axis=1)
y = traindf.y

Xtest = testdf.drop('y', axis=1).drop('ID',axis=1)
ytest = testdf.y


#%%
sel = VarianceThreshold()
vtData = sel.fit_transform(X)

#%%
forest = RandomForestRegressor(n_estimators=500)

forest.fit(X=X,y=y)

for_acc = forest.score(Xtest, ytest)
print('Accuracy of Random Forest Regressor: %s' % '{0:.2%}'.format(for_acc))

#%% Get feature importances for fit RF model

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


#%% # !! Need to Scale Data


#Drop ID before scaling
nmerc = nmerc.drop(['ID'],axis=1)
nmerc_scaled = preprocessing.scale(nmerc)


#%%

 







#-
#--
#---
#----
#-----
#------
#-------
#--------
#---------
#----------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#%%
"""
@title: Implementing Neural Network using Keras in Python

"""
#%%
import os
os.chdir('KerasPython')
#%%
from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(9291)

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
#%%

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#%%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
model.fit(X, Y, nb_epoch=150, batch_size=10)

#%%
scores = model.evaluate(X,Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#%%
