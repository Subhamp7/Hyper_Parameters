# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:54:08 2020

@author: subham
"""
#importing the libraries
import pandas as pd
import numpy  as np
import warnings
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics          import accuracy_score,classification_report,confusion_matrix
warnings.filterwarnings('ignore')

#loading the dataset
dataset=pd.read_csv('heart_data.csv', sep='\t' )
    
#splitting the dataset to independent and dependent sets
dataset_X=dataset.iloc[:,  0:13].values
dataset_Y=dataset.iloc[:, 13:14].values

#scaling the dataset
sc=StandardScaler()
dataset_X=sc.fit_transform(dataset_X)

#splitting data to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(dataset_X, dataset_Y, test_size=0.25 , random_state=0)

#fitting the model
rf=RandomForestClassifier(n_estimators=10)
rf.fit(X_train,Y_train)
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

#Random Search CV'

#creating the grid
n_estimators      = [int(index) for index in np.linspace(start= 200, stop= 2000, num=10)]
max_features      = ['auto', 'sqrt',' log2']
max_depth         = [int(index) for index in np.linspace(10, 1000, 10)]
min_samples_split = [2, 5, 10, 14]
min_samples_leaf  = [1, 2, 4, 6, 8]
criterion         = ['entropy','gini']

random_grid={'n_estimators'       : n_estimators,
               'max_features'     : max_features,
               'max_depth'        : max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
              'criterion'         :criterion}

#fitting the grid to model
rf_1=RandomForestClassifier()
rf_random=RandomizedSearchCV(estimator=rf_1, param_distributions=random_grid,
                             n_iter=100, cv=3, verbose=2, random_state=100, n_jobs=-1)
rf_random.fit(X_train,Y_train)
print("The best params are :\n",rf_random.best_params_)

#fitting the random cv model
model_random=rf_random.best_estimator_
model_random.fit(X_train,Y_train)
pred_1=model_random.predict(X_test)

#calculating accuracy for random cv
print(confusion_matrix(Y_test,pred_1))
print(accuracy_score(Y_test,pred_1))
print(classification_report(Y_test,pred_1))