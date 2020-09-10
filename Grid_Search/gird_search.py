# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:54:08 2020

@author: subham
"""
#importing the libraries
import pandas as pd
import warnings
from sklearn.model_selection       import train_test_split
from sklearn.ensemble              import RandomForestClassifier
from sklearn.model_selection       import GridSearchCV
from sklearn.metrics               import accuracy_score,classification_report,confusion_matrix
warnings.filterwarnings('ignore')

#loading the dataset
dataset=pd.read_csv('bank_data.csv')
    
#splitting the dataset to independent and dependent sets
dataset_X=dataset.iloc[:,  1:11]
dataset_Y=dataset.iloc[:,  11:12]
print(dataset_Y["TARGET CLASS"].value_counts())

#splitting data to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(dataset_X, dataset_Y, test_size=0.3 , random_state=0)

#fitting the model
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)

#predicting
pred=rf.predict(X_test)

#calculating accuracy
print(confusion_matrix(Y_test,pred))
print(accuracy_score(Y_test,pred))
print(classification_report(Y_test,pred))

# The main parameters used by a Random Forest Classifier are:
# 
# - criterion = the function used to evaluate the quality of a split.
# - max_depth = maximum number of levels allowed in each tree.
# - max_features = maximum number of features considered when splitting a node.
# - min_samples_leaf = minimum number of samples which can be stored in a tree leaf.
# - min_samples_split = minimum number of samples necessary in a node to cause node splitting.
# - n_estimators = number of trees in the ensamble.

#grid Search CV'

#creating the grid
n_estimators      = [20,50,80,100,120,150,170,200]
max_depth         = [4,6,8,12,14,16,18,20]

#converting the paramets into a json format
grid_para={'n_estimators': n_estimators,       
           'max_depth'   : max_depth}

#fitting the model
rf_2=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf_2,param_grid=grid_para,cv=5,n_jobs=-1,verbose=2)
grid_search.fit(X_train,Y_train)

#getting the best parameters
best_grid=grid_search.best_estimator_

#fitting into the data and predicting
best_grid.fit(X_train,Y_train)
pred_2=best_grid.predict(X_test)

#validation
print(confusion_matrix(Y_test,pred_2))
print(accuracy_score(Y_test,pred_2))
print(classification_report(Y_test,pred_2))