# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:27:00 2020

@author: subham
"""

#importing the libraries
import pandas as pd
import numpy  as np
import warnings
from sklearn.model_selection       import train_test_split
from sklearn.ensemble              import RandomForestClassifier
from sklearn.metrics               import accuracy_score,classification_report,confusion_matrix
from tpot                          import TPOTClassifier

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

#applying randomized search

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}

#applying the genetic
tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,Y_train)
accuracy = tpot_classifier.score(X_test, Y_test)
print(accuracy)
