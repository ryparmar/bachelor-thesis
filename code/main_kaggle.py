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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
df_test = data_preprocessing.impute_features(df_test, method="median")

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

gc.collect()


"""
PRE-MODELLING PART
train-test split, normalization, shuffling
"""
# reduce dataset size - uncomment for use of SVM
#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=12) 
#
#for idx, test_idx in sss.split(X, y):
#    X = X[idx]
#    y = y[idx]


X_test_test = np.asarray(df_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

mascaler = preprocessing.MaxAbsScaler()
X_train = mascaler.fit_transform(X_train)
X_test = mascaler.transform(X_test)
X_test_test = mascaler.transform(X_test_test)

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
              "max_features": ["sqrt", "auto"],
              "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 15],
              "min_samples_leaf": [2, 4, 6, 8, 10, 12, 14, 15],
              "n_estimators": [20, 40, 80, 150],
              "bootstrap": [True],
              "criterion": ["gini", "entropy"]}

parameters_lr = {"C": np.arange(1, 31)}


"""
Upsampling
"""
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4)
X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
X_res_tr, y_res_tr= sm.fit_sample(X_tr, y_tr)
X_res_val, y_res_val = sm.fit_sample(X_val, y_val)
X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
print(pd.Series(y_res_train).value_counts())
print(pd.Series(y_res_val).value_counts())


rf = models.RF()
rf_opt, rf_opt_params = modelling.find_hyperparams(rf, parameters_rf,\
                                                   X_res_train,\
                                                   y_res_train, \
                                                   search_method="randomized",\
                                                   n_iter=50)
rf_res_score = modelling.evaluation(rf_opt, X_res_test, y_res_test, rf_opt.predict(X_res_test))
rf_score = modelling.evaluation(rf_opt, X_test, y_test, rf_opt.predict(X_test))

from sklearn.ensemble import RandomForestClassifier
ls = RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=20, 
                            max_features="sqrt", min_samples_leaf=4, min_samples_split=4,
                            n_estimators=200, oob_score=True)
ls.fit(X_res_train, y_res_train)

#feature selection
#imp = ls.feature_importances_ > 0.01
#imp
#feat=[]
#for i in range(0, features.shape[0]):
#    if imp[i] == True:
#        feat.append(features[i])
#        print (features[i])

ls_score = modelling.evaluation(ls, X_test, y_test, ls.predict(X_test))
ls_res_score = modelling.evaluation(ls, X_res_test, y_res_test, ls.predict(X_res_test))
print(ls.oob_score_)

ls_pred = ls.predict_proba(X_test_test)[:,1]

kaggle = pd.DataFrame(data=ls_pred, index=range(1, len(df_test.index)+1), columns=["Probability"])
kaggle.index.name = "Id"
kaggle.to_csv("bachelor-thesis/kaggle.csv")

nn = models.NN_15(X_train)
nn.fit(X_res_train, y_res_train, batch_size=512, epochs=15, validation_data=(X_res_val, y_res_val))

nn_res_pred = nn.predict(X_res_test)
for i in range(0, len(X_res_test)):
    if(nn_res_pred[i] > 0.5):
        nn_res_pred[i]=1
    else:
        nn_res_pred[i]=0
nn_res_score = modelling.evaluation(nn, X_res_test, y_res_test, nn_res_pred)

nn_pred = nn.predict(X_test)
for i in range(0, len(X_test)):
    if(nn_pred[i] > 0.5):
        nn_pred[i]=1
    else:
        nn_pred[i]=0
nn_score = modelling.evaluation(nn, X_test, y_test, nn_pred)

#nn_pred_proba = nn.predict_proba(df_test.values)
nn_pred_proba = nn.predict(df_test.values)
kaggle = pd.DataFrame(data=nn_pred_proba, index=range(1, len(df_test.index)+1), columns=["Probability"])
kaggle.index.name = "Id"
kaggle.to_csv("bachelor-thesis/kaggle.csv")



lr = models.LR()
lr_opt, lr_opt_params = modelling.find_hyperparams(lr, parameters_lr, \
                                                   X_res_train, \
                                                   y_res_train, \
                                                   search_method="gridsearch", \
                                                   cv=10)
lr_res_score = modelling.evaluation(lr_opt, X_res_test, y_res_test, lr_opt.predict(X_res_test))
lr_score = modelling.evaluation(lr_opt, X_test, y_test, lr_opt.predict(X_test))


from sklearn import svm
cl = svm.LinearSVC(C=100) 
cl.fit(X_res_train, y_res_train)
svm_res_score = modelling.evaluation(cl, X_res_test, y_res_test, cl.predict(X_res_test))
svm_score = modelling.evaluation(cl, X_test, y_test, cl.predict(X_test))



#svm = models.SVM()
#svm_opt, svm_opt_params = modelling.find_hyperparams(svm, parameters_svm, \
#                                                   X_train, \
#                                                   y_train, \
#                                                   search_method="gridsearch", \
#                                                   cv=10,)
#svm_score = modelling.evaluation(svm_opt, X_test, y_test, svm_opt.predict(X_test))


gc.collect()