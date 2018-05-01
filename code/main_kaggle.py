#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAGGLE DATASET
--------------
INITIALIZATION
"""  
import sys, gc
sys.path
path = '/home/spaceape/bachelor_thesis/code'
if path not in sys.path:
    sys.path.append(path)

import data_preprocessing, models, modelling
from scipy.stats import randint as sp_randint
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import sklearn.metrics
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = '/home/spaceape/bachelor_thesis/data/'


"""
DATA WRANGLING
"""
df_train, df_test = data_preprocessing.load_gsc_data(path)

df_train = df_train.drop(["Unnamed: 0"], 1)
df_test = df_test.drop(["Unnamed: 0", "SeriousDlqin2yrs"], 1)

df_train = data_preprocessing.impute_features(df_train, method="median")

df_train = data_preprocessing.create_features(df_train)
df_test = data_preprocessing.create_features(df_test)

X, y = data_preprocessing.set_structure(df_train, y="SeriousDlqin2yrs")
df_train = df_train.drop(["SeriousDlqin2yrs"], 1)

cat, num, cat_idx, num_idx = data_preprocessing.get_features(df_train)
features = pd.Series(index=(cat_idx + num_idx), data=(cat + num)).sort_index()

"""
uncomment to apply feature selection
"""
# return two series with selected features using select K best
#chi2_score = modelling.best_features(X, y, features, statistics="chi2")
#X = modelling.feature_selection(X, y, "chi2", k=5)


del df_train
del df_test
gc.collect()


"""
PRE-MODELLING PART
train-test split, normalization, shuffling
"""
# reduce dataset size - uncomment for use of SVM
#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=12) 

#for idx, test_idx in sss.split(X, y):
#    X = X[idx]
#    y = y[idx]

K = 10
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state = 10)

for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


mascaler = preprocessing.MaxAbsScaler()
X_train = mascaler.fit_transform(X_train)
X_test = mascaler.transform(X_test)
    
# shuffle
idx = np.arange(len(X_train))     
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]



"""
MODELLING PART
"""
# define grid of hyperparameters
parameters_svm = {"kernel" : ["linear"],
                  "C" : [1, 10, 30, 50, 60, 80, 100, 200, 300],
                  "gamma" : [0.01, 0.001]}
    
parameters_rf = {"max_depth": [3, 5, 8, 10, 12, 14, 16, 18, 20],
              "max_features": ["sqrt", "auto"],#[1, 3, 5, 7, 9, 11],
              "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 15],
              "min_samples_leaf": [2, 4, 6, 8, 10, 12, 14, 15],
              "n_estimators": [10, 18, 24, 36],
              "bootstrap": [True],
              "criterion": ["gini", "entropy"]}

parameters_lr = {"C": np.arange(1, 31)}

rf = models.RF()
rf_opt, rf_opt_params = modelling.find_hyperparams(rf, parameters_rf,\
                                                   X_train,\
                                                   y_train, \
                                                   search_method="randomized",\
                                                   n_iter=50)
rf_score = modelling.evaluation(rf_opt, X_test, y_test, rf_opt.predict(X_test))


"""
Upsampling technique
"""
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=50)
#print(pd.Series(y_res).value_counts())
nn = models.NN_512(X_train)
X_res, y_res = sm.fit_sample(X_train, y_train)
nn.fit(X_res, y_res, batch_size=512, epochs=5, validation_split=0.3)
nn_pred = nn.predict(X_test)
for i in range(0, len(X_test)):
    if(nn_pred[i] > 0.5):
        nn_pred[i]=1
    else:
        nn_pred[i]=0  
nn_score = modelling.evaluation(nn, X_test, y_test, nn_pred)


"""
class weight technique
"""
nn = models.NN_15(X_train)
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
nn.fit(X_train, y_train, batch_size=512, epochs=5, validation_split=0.3, class_weight=class_weight)
nn_pred = nn.predict(X_test)
for i in range(0, len(X_test)):
    if(nn_pred[i] > 0.3):
        nn_pred[i]=1
    else:
        nn_pred[i]=0
nn_score = modelling.evaluation(nn, X_test, y_test, nn_pred)

lr = models.LR()
lr_opt, lr_opt_params = modelling.find_hyperparams(lr, parameters_lr, \
                                                   X_train, \
                                                   y_train, \
                                                   search_method="gridsearch", \
                                                   cv=10)
lr_score = modelling.evaluation(lr_opt, X_test, y_test, lr_opt.predict(X_test))


from sklearn import svm
cl = svm.SVC(C=50, gamma=0.01, cache_size=6000, kernel="linear") 
cl.fit(X_train, y_train)
svm_score = modelling.evaluation(cl, X_test, y_test, cl.predict(X_test))
#50, 0.001
# 1, 0.01



#svm = models.SVM()
#svm_opt, svm_opt_params = modelling.find_hyperparams(svm, parameters_svm, \
#                                                   X_train, \
#                                                   y_train, \
#                                                   search_method="gridsearch", \
#                                                   cv=10,)
#svm_score = modelling.evaluation(svm_opt, X_test, y_test, svm_opt.predict(X_test))


gc.collect()