# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:46:02 2020

@author: subham
"""

#importing the libraries
import pandas as pd
import optuna
import sklearn.svm
import warnings
from sklearn.model_selection       import train_test_split
from sklearn.ensemble              import RandomForestClassifier
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

#applying optuna
def objective(trial):

    classifier = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])
    
    if classifier == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        max_depth = int(trial.suggest_float('max_depth', 10, 100, log=True))

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
    else:
        c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
        
        clf = sklearn.svm.SVC(C=c, gamma='auto')

    return sklearn.model_selection.cross_val_score(
        clf,X_train,Y_train, n_jobs=-1, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

study.best_params

rf=RandomForestClassifier(n_estimators=330,max_depth=30)
rf.fit(X_train,Y_train)

Y_pred=rf.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))