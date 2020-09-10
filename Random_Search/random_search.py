# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:57:17 2020

@author: subham
"""
#importing the libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection       import train_test_split
from sklearn.ensemble              import RandomForestClassifier
from sklearn.model_selection       import RandomizedSearchCV
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


#random search

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start =100, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

rf_1=RandomForestClassifier()
randomcv=RandomizedSearchCV(estimator=rf_1,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
randomcv.fit(X_train,Y_train)

#getting the best parameters
best_grid=randomcv.best_estimator_

#fitting into the data and predicting
best_grid.fit(X_train,Y_train)
pred_2=best_grid.predict(X_test)

#validation
print(confusion_matrix(Y_test,pred_2))
print(accuracy_score(Y_test,pred_2))
print(classification_report(Y_test,pred_2))

