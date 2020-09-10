# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:44:16 2020

@author: subham
"""

#importing the libraries
import pandas as pd
import warnings
from sklearn.model_selection       import train_test_split
from sklearn.ensemble              import RandomForestClassifier
from sklearn.model_selection       import cross_val_score
from sklearn.metrics               import accuracy_score,classification_report,confusion_matrix
from hyperopt                      import hp,fmin,tpe,STATUS_OK,Trials

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

#applying bayesian optimization
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])}

def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
    accuracy = cross_val_score(model, X_train, Y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)

#for space data with choice we need to make another dict
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}

#applying the RFC with best parameters
trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(X_train,Y_train)
predictionforest = trainedforest.predict(X_test)
print(confusion_matrix(Y_test,predictionforest))
print(accuracy_score(Y_test,predictionforest))
print(classification_report(Y_test,predictionforest))
acc5 = accuracy_score(Y_test,predictionforest)
