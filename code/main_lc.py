#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
LENDING CLUB DATASET
--------------------
INITIALIZATION
"""  
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sys, gc

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



path = '/home/spaceape/bachelor_thesis/code'
if path not in sys.path:
    sys.path.append(path)
import data_preprocessing, models, modelling
path = '/home/spaceape/bachelor_thesis/data'


""" 
DATA WRANGLING
"""
df = data_preprocessing.load_LC_data(path, sample=True)

# keeping only best 4 categorical variable according to select-k best chi2 test
remove_cat = ['grade', 'sub_grade', 'last_pymnt_d', 'last_credit_pull_d',
              "zip_code", "earliest_cr_line", "issue_d", "pymnt_plan", "purpose", "title", 
              "hardship_flag", "application_type", "addr_state"]
df = data_preprocessing.remove_cat(df, remove_cat)
# sorted by importance 
# "verification_status", "home_ownership", "emp_title", "initial_list_status"


# feature selected using random forest feature importances method
feat = ['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','pub_rec',
        'initial_list_status','out_prncp',
        'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
        'total_rec_late_fee','recoveries', "verification_status", "home_ownership",
        'collection_recovery_fee', 'last_pymnt_amnt','total_rev_hi_lim','avg_cur_bal',
        "loan_status"]
df = df[feat]


#from scipy import stats
#df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


df = data_preprocessing.drop_nans_col_row(df)

df = data_preprocessing.manual_encoding(df)

df = data_preprocessing.drop_nans_all(df)

cat, num, cat_idx, num_idx = data_preprocessing.get_features(df)
features = pd.Series(index=(cat_idx + num_idx), data=(cat + num)).sort_index()

df, emp = data_preprocessing.feature_handling(df)

#print(df["emp_title"].value_counts())



"""
uncomment to apply feature selection
"""

# finding best categorical features
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#for feature in cat:
#    bf = le.fit_transform(df[feature])
#    df[feature] = bf
#del bf
#
#best = modelling.best_features(df[cat], df["loan_status"], cat)


df, X_cat = data_preprocessing.categorical_encoding(df, cat, one_hot_encoding=True)

#
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder(sparse=True, dtype=np.uint8)    
#for feature in cat:
#    tmp1 = ohe.fit_transform(df[feature].reshape(len(df), 1))
#    df[feature]=tmp1

# return X, y as ndarrays
X, y = data_preprocessing.set_structure(df, y="loan_status")



# reduce dataset size - uncomment for use of SVM
#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=12) 
#for idx, test_idx in sss.split(X, y):
#    X = X[idx]
#    y = y[idx]
#    if X_cat != []:
#        for i in X_cat:
#            X_cat[i] = X_cat[[i][idx]]



X_cat = np.concatenate(X_cat, axis=1)

num_idx.remove(71)

X = np.concatenate((X[:, num_idx], X_cat), axis=1)

K = 10
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state = 10)
for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.preprocessing import MinMaxScaler
mascaler = preprocessing.MaxAbsScaler()
mascaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X_train[:, num_idx] = mascaler.fit_transform(X_train[:, num_idx])
X_test[:, num_idx] = mascaler.transform(X_test[:, num_idx])

# shuffle
idx = np.arange(len(X_train))     
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]

## reduce dataset size - uncomment for use of SVM
#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=12) 
#
#for idx, test_idx in sss.split(X_train, y_train):
#    X_train = X_train[idx]
#    y_train = y_train[idx]


"""
MODELLING PART
"""
# define grid of hyperparameters
parameters_svm = {"kernel" : ["linear", "rbf"],
                  "gamma" : [0.01, 0.001],
                  "C" : [1, 5, 10, 20, 40, 100, 1000]
                  }

parameters_rf = {"max_depth": [3, 5, 8, 10, 12, 14, 16, 18, 20],
              "max_features": ["sqrt", "auto"],
              "min_samples_split": [2, 4, 6, 8, 10, 12, 14, 15],
              "min_samples_leaf": [2, 4, 6, 8, 10, 12, 14, 15],
              "n_estimators": [10, 18, 24, 36],
              "bootstrap": [True],
              "criterion": ["gini", "entropy"]}

parameters_lr = {"C": [1, 3, 6, 9, 15 , 20, 25, 30]}

parameters_lin_svm = {"C" : [1, 5, 10, 20, 40, 100, 1000]}


rf = models.RF()
rf_opt, rf_opt_params = modelling.find_hyperparams(rf, parameters_rf, X_train, y_train, \
                                                   search_method="randomized", \
                                                   n_iter = 10)
rf_score = modelling.evaluation(rf_opt, X_test, y_test, rf_opt.predict(X_test))

#from sklearn.ensemble import RandomForestClassifier
#ls = RandomForestClassifier(bootstrap=True, criterion="gini", max_depth=20, 
#                            max_features="sqrt", min_samples_leaf=8, min_samples_split=2,
#                            n_estimators=24)
#ls.fit(X_train, y_train)
#imp = ls.feature_importances_ > 0.001
#feat=[]
#for i in range(0, features.shape[0]):
#    if imp[i] == True:
#        feat.append(features[i])
#        print (features[i])

#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(15,), activation="relu", solver="adam")
#mlp.fit(X_train, y_train)



"""
class weight technique
"""
nn = models.NN_512(X_train)
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
nn.fit(X_train, y_train, batch_size=512, epochs=5, validation_split=0.3, class_weight=class_weight)
nn_pred = nn.predict(X_test)
print(nn_pred)
nn_pred1 = nn_pred.copy()
for i in range(0, len(X_test)):
    if(nn_pred1[i] > 0.5):
        nn_pred[i]=1
    else:
        nn_pred[i]=0
nn_score = modelling.evaluation(nn, X_test, y_test, nn_pred)

"""
Upsampling technique
"""
nn = models.NN_512(X_train)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12) #random_state=50
X_res, y_res = sm.fit_sample(X_train, y_train)
print(pd.Series(y_res).value_counts())

nn.fit(X_res, y_res, batch_size=512, epochs=10, validation_split=0.3)
nn_pred = nn.predict(X_test)
for i in range(0, len(X_test)):
    if(nn_pred[i] > 0.5):
        nn_pred[i]=1
    else:
        nn_pred[i]=0     
nn_score = modelling.evaluation(nn, X_test, y_test, nn_pred)



# opt C=30
lr = models.LR()
lr_opt, lr_opt_params = modelling.find_hyperparams(lr, parameters_lr, \
                                                   X_train, \
                                                   y_train, \
                                                   search_method="gridsearch", \
                                                   cv=10)
lr_score = modelling.evaluation(lr_opt, X_test, y_test, lr_opt.predict(X_test))


svm = models.linSVM()
svm_opt, svm_opt_params = modelling.find_hyperparams(svm, parameters_lin_svm, \
                                                     X_train,\
                                                     y_train, search_method="gridsearch")
svm_score = modelling.evaluation(svm, X_test, y_test, svm.predict(X_test))

from sklearn import svm
svm = svm.SVC(class_weight="balanced", C=1000)
svm.fit(X_train, y_train)
svm_score = modelling.evaluation(svm, X_test, y_test, svm.predict(X_test))

#svm_opt, svm_opt_params = modelling.find_hyperparams(svm, parameters_svm, \
#                                                     X_train,\
#                                                     y_train, search_method="gridsearch")
#svm_score = modelling.evaluation(svm_opt, X_test, y_test, svm_opt.predict(X_test))




gc.collect()
