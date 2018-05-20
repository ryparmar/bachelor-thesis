#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:10:50 2018

@author: spaceape

Below script provide basic data wrangling functions with dataframe. 
notice the features are in dataframe stored as columns, thus all the handling with feature is named
col; similarly for observations stored in rows.

row = observation
col = feature

"""


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Imputer


# load Lending Club data to pandas dataframe
def load_LC_data(data_folder, sample=True, rejected=False, low_memory=False):
    if rejected == True:
        if sample == True:
            df_a = pd.read_csv(data_folder + "lending-club/sample_acc.csv")
            df_r = pd.read_csv(data_folder + "/lending-club/sample_rej.csv")
        else:
            df_a = pd.read_csv(data_folder + "/lending-club/accepted_2007_to_2017.csv")
            df_r = pd.read_csv(data_folder + "/lending-club/sample_rej.csv")
            return df_a, df_r
    else:
        if sample == True:
            df_a = pd.read_csv(data_folder + "/lending-club/sample_acc.csv")
        else:
            df_a = pd.read_csv(data_folder + "/lending-club/accepted_2007_to_2017.csv")
        return df_a
            
    
# load Kaggle data to pandas dataframe
def load_gsc_data(data_folder):
    df_train = pd.read_csv(data_folder + "gsc/cs-training.csv")
    df_test = pd.read_csv(data_folder + "gsc/cs-test.csv")
    return df_train, df_test


#count number of nans within each feature (col) and each observation (row)
def count_nans(df):
    nans_col = pd.Series(df.isnull().sum(axis=0), name="nans")
    nans_col = nans_col.sort_values(ascending=False)
    nans_row = pd.Series(df.isnull().sum(axis=1), name="nans")
    nans_row = nans_row.sort_values(ascending=False)
    return nans_col, nans_row


# impute missing values by chosen method
def impute_features(df, method=str):
    nans_col, nans_row = count_nans(df)
    if method == "median":
        imp = Imputer(missing_values=np.nan, strategy='median', axis=0)
        for feature in df[:]:
            if feature != "RepaymentHistory":
                if (nans_col[feature] > (0*len(df))):
                    print (feature, nans_col[feature])
    elif method == "most_frequent":
        imp = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0)
        for feature in df[:]:
            if feature != "RepaymentHistory":
                if (nans_col[feature] > (0*len(df))):
                    print (feature, nans_col[feature])
    else:
        print("DATA WAS NOT IMPUTED")
    return pd.DataFrame(imp.fit_transform(df), index=df.index, columns=df.columns)


# drop the features(columns) containing more than x% of NaNs given in drop_col parameter
# and observations (rows) with more than y NaNs given in drop_row parameter 
def drop_nans_col_row(df, drop_col=0.2, drop_row=30):
    nans_col, nans_row = count_nans(df)
    
    # features (cols)
    for feature in df:
        if (nans_col[feature] > (drop_col*len(df)) or (feature == "Unnamed: 0") or \
            (len(df[feature].unique()) == 1)):
            print (feature, nans_col[feature])
            df = df.drop(feature, 1)
            
    nans_col, nans_row = count_nans(df)
    
    # observations (rows)        
    del_rows = nans_row > drop_row
    del_rows = del_rows[del_rows==True].index
    df = df.drop(del_rows, axis=0)
    
    df = df.reset_index(drop=True)
    
    return df


def drop_nans_all(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


# create features in kaggle dataset
def create_features(df):
    if "RepaymentHistory" not in df.columns:
        df["RepaymentHistory"] = pd.Series(index=df.index)
        df["RepaymentHistory"] = df["NumberOfTime30-59DaysPastDueNotWorse"] + \
        df["NumberOfTime60-89DaysPastDueNotWorse"]*2 + \
        df["NumberOfTime60-89DaysPastDueNotWorse"]*3
        return df
    else:
        print("FEATURE IS ALREADY IN DATAFRAME")

# manually replace some of the feature values
def manual_encoding(df):
    # encoding emp_length from Lending Club dataset
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].replace(["< 1 year", "1 year", "2 years", "3 years",\
          "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10 years",\
          "10+ years"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    else:
        print("feature not in dataframe columns")
    
    # encoding dependent variable of Lending Club dataset    
    if "loan_status" in df.columns:
        stat_used = ["Current", "Fully Paid", "Default", "Charged Off", "Late (31-120)", "Late (16-30)", "In Grace Period"]
        default = [0, 0, 1, 1, 1, 0, 0]
        stat_notused = []
        nans = []

        for i in (df["loan_status"].unique()):   
            if i not in stat_used:
                stat_notused.append(i)
                nans.append(np.nan)

        df["loan_status"] = df["loan_status"].replace(stat_notused, nans)
        temp = df["loan_status"] == np.nan
        temp = temp[temp==True].index
        df = df.drop(temp, axis=0)
        df["loan_status"] = df["loan_status"].replace(stat_used, default)
        df = df.reset_index(drop=True)
        
        df["term"] = df["term"].replace([" 36 months", " 60 months"], [36, 60])
    return df

def remove_feature(df, feature_remove):
    if feature_remove in df.columns:
        df = df.drop(feature_remove, 1)
    return df

# get lists of categorical and numerical features
def get_features(df):
    categorical = list()
    numerical = list()
    categorical_idx = list()
    numerical_idx = list()

    for feature in df:
        if (type(df[feature][0]) == type("string") and feature != "loan_status" and feature != "SeriousDlqin2yrs"):
            categorical.append(feature)
            categorical_idx.append(df.columns.get_loc(feature))
#            print ("categorical", feature, df.columns.get_loc(feature))
        elif (type(df[feature][0]) != type("string") and feature != "loan_status" and feature != "SeriousDlqin2yrs"):
            numerical.append(feature)
            numerical_idx.append(df.columns.get_loc(feature))
#            print ("numerical", feature, df.columns.get_loc(feature))
    return categorical, numerical, categorical_idx, numerical_idx


def get_features_idx(df, cat, num):
    categorical_idx = list()
    numerical_idx = list()

    for feature in cat:
        if (feature != "loan_status" and feature != "SeriousDlqin2yrs"):
            categorical_idx.append(df.columns.get_loc(feature))
            #print (feat, nans_fa[feat])
    for feature in num:
        if (feature in num and feature != "loan_status" and feature != "SeriousDlqin2yrs"):
            numerical_idx.append(df.columns.get_loc(feature))
            #print (feat, nans_fa[feat])
    return categorical_idx, numerical_idx


# proces the empl_title (LC dataset) in order to reduce dimension
def feature_handling(df):
    # emp_title encoding
    emp = list()
    if "emp_title" in df.columns:
        for i in df.index:
            if type(df["emp_title"][i]) == type("string"):
                df.set_value(i, "emp_title", df["emp_title"][i].lower())
                if df["emp_title"][i] == "rn":
                    df.set_value(i, "emp_title", "registered nurse")
                elif "assistant" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "assistant")
                elif "manager" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "manager")
                elif "analyst" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "analyst")
                elif "director" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "director")
                elif "engineer" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "engineer")
                elif "owner" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "owner")
                elif "president" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "president")
                elif "supervisor" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "supervisor")
                elif "technician" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "technician")
                elif "consultant" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "consultant")
                elif "mechanic" in df["emp_title"][i]:
                    df.set_value(i, "emp_title", "mechanic")
        
        tmp = df["emp_title"].value_counts()        
        for i in tmp.index:
            if tmp[i] > 100:
                emp.append(i)
        emp.append("others")
        
        for i in df.index:
            if df["emp_title"][i] not in emp:
                df.set_value(i, "emp_title", "others")
    
    # title encoding
    emp = list()
    if "title" in df.columns:
        for i in df.index:
             if type(df["title"][i]) == type("string"):
                df.set_value(i, "title", df["title"][i].lower())
                if "consolidate" in df["emp_title"][i]:
                    df.set_value(i, "title", "debt consolidation")
                elif "consolidation" in df["emp_title"][i]:
                    df.set_value(i, "title", "debt consolidation")
            
        tmp = df["title"].value_counts()        
        for i in tmp.index:
            if tmp[i] > 100:
                emp.append(i)
        
        for i in df.index:
            if df["title"][i] not in emp:
                df.set_value(i, "title", "other")
                       
    return df, emp


# provide label encoding and one_hot_encoding for categorical variables
def categorical_encoding(df, categorical, one_hot_encoding=True):
    # label encoding categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for feature in categorical:
        #print (feat, len(df_a[feat].unique()), df_a[feat].unique())
        tmp = le.fit_transform(df[feature])
        df[feature] = tmp
        
    X_cat = []
    # one-hot-encoding categorical variables    
    if one_hot_encoding == True:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False, dtype=np.uint8)    
        for feature in categorical:
            tmp1 = ohe.fit_transform(df[feature].reshape(len(df), 1))
            X_cat.append(tmp1)
            
    return df, X_cat


# set the structure of dataset to ndarray and create ndarray with numerical features


# 
def set_structure(df, y=str):
    if y == "loan_status" and "loan_status" in df.columns:
        y = np.asarray(df["loan_status"], dtype=np.int8)
        X = np.asarray(df.drop("loan_status", 1))
    elif y == "SeriousDlqin2yrs" and "SeriousDlqin2yrs" in df.columns:
        y = np.asarray(df["SeriousDlqin2yrs"], dtype=np.int8)
        X = np.asarray(df.drop("SeriousDlqin2yrs", 1))
    else:
        print("SOMETHING WENT WRONG")
    return X, y
        

# provide scaling of numerical features; return scaled ndarray in range <1;1>
def feature_scaling_fit_transform(X):
    from sklearn import preprocessing
    mascaler = preprocessing.MaxAbsScaler()
    X = mascaler.fit_transform(X)
    return X

def feature_scaling_transform(X):
    from sklearn import preprocessing
    mascaler = preprocessing.MaxAbsScaler()
    X = mascaler.transform(X)
    return X


# concatenate inputed ndarrays/ndarray; return concatenated ndarray
def concatenate(X_cat, X_num):
    if len(X_num) == 0 and type(X_cat) == list:
        XX = np.concatenate(X_cat, axis=1)
    elif type(X_cat) == list:
        XX = X_cat
        for i in range(X_num.shape[1]):
            XX.append(X_num[:, i])
    else:
        XX = np.concatenate((X_cat, X_num), axis=1)
    return XX
