#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
LENDING CLUB DATASET
--------------------
INITIALIZATION
"""  
from sklearn.model_selection import train_test_split
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
#
features = ['loan_amnt', 'term', 'installment',
        'emp_title', 'home_ownership', 'emp_length',
       'annual_inc', 'verification_status', 'issue_d', 'loan_status',
       'purpose', 'title', 'addr_state', 'dti', 'delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'mths_since_last_major_derog',
       'application_type', 'annual_inc_joint', 'dti_joint',
       'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt',
       'tot_cur_bal', 'open_acc_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl',
       'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
       'bc_util', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
       'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
       'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
       'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
       'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
       'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
       'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
       'total_il_high_credit_limit']

# FEATURES SELECTED
features = ['loan_amnt', 'term', 'emp_title', 'home_ownership', 'annual_inc', 'issue_d', 'purpose',
 'inq_last_6mths', 'open_acc', 'revol_bal', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'delinq_amnt', 
 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc',
 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_tl_30dpd', "loan_status"]

#num = ['loan_amnt', 'term', 'installment',
#       'emp_length',
#       'annual_inc',
#       'dti', 'delinq_2yrs',
#       'inq_last_6mths', 'mths_since_last_delinq',
#       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
#       'revol_util', 'total_acc', 'mths_since_last_major_derog',
#       'annual_inc_joint', 'dti_joint',
#       'acc_now_delinq', 'tot_coll_amt',
#       'tot_cur_bal', 'open_acc_6m', 'open_il_12m',
#       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
#       'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl',
#       'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
#       'bc_util', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
#       'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
#       'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
#       'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
#       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
#       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
#       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
#       'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
#       'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
#       'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
#       'total_il_high_credit_limit']
#
#cat = ["emp_title", "title", "home_ownership", "verification_status", "issue_d", "purpose",
#       "addr_state", "application_type", "verification_status_joint"]



# feature selected using random forest feature importances method
#feat = ['loan_amnt','funded_amnt','funded_amnt_inv','term','installment','pub_rec',
#        'initial_list_status','out_prncp',
#        'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
#        'total_rec_late_fee','recoveries', "verification_status", "home_ownership",
#        'collection_recovery_fee', 'last_pymnt_amnt','total_rev_hi_lim','avg_cur_bal',
#        "loan_status"]


#df = df[feat]


#from scipy import stats
#df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


df = data_preprocessing.drop_nans_col_row(df)

df = data_preprocessing.manual_encoding(df)
  
df = data_preprocessing.drop_nans_all(df)
for feature in df.columns:
    if feature not in features:
        df = df.drop(feature, 1)

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

#for c in cat:
#    df[cat] = df[cat].replace([np.nan], ["NaN"])

df, X_cat = data_preprocessing.categorical_encoding(df, cat, one_hot_encoding=True)

for i in range(0, len(X_cat)):
    print(cat[i], " ", X_cat[i].shape[1])

# return X, y as ndarrays
X, y = data_preprocessing.set_structure(df, y="loan_status")



# reduce dataset size - uncomment for use of SVM
#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=12) 
#for idx, test_idx in sss.split(X, y):
#    X = X[idx]
#    y = y[idx]
#    if X_cat != []:
#        for i in X_cat:
#            X_cat[i] = X_cat[[i][idx]]



X_cat = np.concatenate(X_cat, axis=1)

num_idx.pop()

X = np.concatenate((X[:, num_idx], X_cat), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

from sklearn.preprocessing import MinMaxScaler
mascaler = preprocessing.MaxAbsScaler()
mascaler = MinMaxScaler(copy=True, feature_range=(0, 1))

# scaling for RF, LR, SVM
X_train[:, 0:len(num_idx)] = mascaler.fit_transform(X_train[:, 0:len(num_idx)])
X_test[:, 0:len(num_idx)] = mascaler.transform(X_test[:, 0:len(num_idx)])

# scaling also validation set for NN
#X_tr[:, 0:len(num_idx)] = mascaler.fit_transform(X_tr[:, 0:len(num_idx)])
#X_val[:, 0:len(num_idx)] = mascaler.transform(X_val[:, 0:len(num_idx)])
#X_test[:, 0:len(num_idx)] = mascaler.transform(X_test[:, 0:len(num_idx)])

# shuffle
idx = np.arange(len(X_train))     
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]

## reduce dataset size
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=12) 

for idx, test_idx in sss.split(X_train, y_train):
    X_train = X_train[idx]
    y_train = y_train[idx]


"""
MODELLING PART
"""
# define grid of hyperparameters
parameters_svm = {"kernel" : ["linear"],
                  "gamma" : [0.01, 0.001],
                  "C" : [1, 5, 10, 20, 40, 100, 1000]
                  }

parameters_rf = {"max_depth": [3, 5, 8, 10, 12, 16, 20],
              "max_features": ["sqrt", "auto"],
              "min_samples_split": [2, 4, 6, 8, 10, 12],
              "min_samples_leaf": [2, 4, 6, 8, 10, 12],
              "n_estimators": [20, 40, 100, 200],
              "bootstrap": [True],
              "criterion": ["gini", "entropy"]}

parameters_lr = {"C": [1, 3, 6, 9, 15 , 20, 25, 30]}

parameters_lin_svm = {"C" : [1, 5, 10, 20, 40, 100, 1000]}

"""
Upsampling
"""
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=8)
X_res_train, y_res_train = sm.fit_sample(X_train, y_train)
X_res_tr, y_res_tr= sm.fit_sample(X_tr, y_tr)
X_res_val, y_res_val = sm.fit_sample(X_val, y_val)
X_res_test, y_res_test = sm.fit_sample(X_test, y_test)
print(pd.Series(y_res_train).value_counts())
print(pd.Series(y_res_val).value_counts())


rf = models.RF()
rf_opt, rf_opt_params = modelling.find_hyperparams(rf, parameters_rf, X_res_train, y_res_train, \
                                                   search_method="randomized", \
                                                   n_iter = 50)
rf_score = modelling.evaluation(rf_opt, X_test, y_test, rf_opt.predict(X_test))
rf_res_score = modelling.evaluation(rf_opt, X_res_test, y_res_test, rf_opt.predict(X_res_test))

from sklearn.ensemble import RandomForestClassifier
ls = RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=10, 
                            max_features="sqrt", min_samples_leaf=2, min_samples_split=4,
                            n_estimators=200)
ls.fit(X_res_train, y_res_train)
imp = ls.feature_importances_ > 0.01
imp
feat=[]
for i in range(0, features.shape[0]):
    if imp[i] == True:
        feat.append(features[i])
        print (features[i])

ls_score = modelling.evaluation(ls, X_test, y_test, ls.predict(X_test))
ls_res_score = modelling.evaluation(ls, X_res_test, y_res_test, ls.predict(X_res_test))


nn = models.NN_512(X_res_train)
nn.fit(X_res_train, y_res_train, batch_size=256, epochs=15, validation_split=0.2)#, validation_data=(X_res_val, y_res_val))
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


nn = models.NN_15(X_res_train)
nn.fit(X_res_train, y_res_train, batch_size=512, epochs=20, validation_split=0.2)#, validation_data=(X_res_val, y_res_val))
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



lr = models.LR()
lr_opt, lr_opt_params = modelling.find_hyperparams(lr, parameters_lr, \
                                                   X_res_train, \
                                                   y_res_train, \
                                                   search_method="gridsearch", \
                                                   cv=10)

from sklearn.linear_model import LogisticRegression
lr_opt = LogisticRegression(C=30)
lr_opt.fit(X_res_train, y_res_train)
lr_res_score = modelling.evaluation(lr_opt, X_res_test, y_res_test, lr_opt.predict(X_res_test))
lr_score = modelling.evaluation(lr_opt, X_test, y_test, lr_opt.predict(X_test))



#svm = models.linSVM()
#svm_opt, svm_opt_params = modelling.find_hyperparams(svm, parameters_lin_svm, \
#                                                     X_res_train,\
#                                                     y_res_train, search_method="gridsearch")
#svm_score = modelling.evaluation(svm, X_res_test, y_res_test, svm.predict(X_res_test))

from sklearn import svm
svm = svm.LinearSVC(C=10)
#svm = svm.SVC(C=1)
svm.fit(X_res_train, y_res_train)
svm_res_score = modelling.evaluation(svm, X_res_test, y_res_test, svm.predict(X_res_test))
svm_score = modelling.evaluation(svm, X_test, y_test, svm.predict(X_test))





gc.collect()
