#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:46:20 2018

@author: spaceape
"""
import numpy as np
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding

# testing the embedding
#def NN_embed(X_train, num_idx):
#    models = []
#
#    model_cat_01 = Sequential()
#    model_cat_01.add(Embedding(3, 2, input_length=1))
#    model_cat_01.add(Reshape(target_shape=(2, )))
#    models.append(model_cat_01)
#
#    model_cat_02 = Sequential()
#    model_cat_02.add(Embedding(8, 4, input_length=1))
#    model_cat_02.add(Reshape(target_shape=(4, )))
#    models.append(model_cat_02)
#
#    model_cat_03 = Sequential()
#    model_cat_03.add(Embedding(36, 18, input_length=1))
#    model_cat_03.add(Reshape(target_shape=(18, )))
#    models.append(model_cat_03)
#
#    model_cat_04 = Sequential()
#    model_cat_04.add(Embedding(31938, 50, input_length=1))
#    model_cat_04.add(Reshape(target_shape=(50, )))
#    models.append(model_cat_04)
#
#    model_cat_05 = Sequential()
#    model_cat_05.add(Embedding(7, 4, input_length=1))
#    model_cat_05.add(Reshape(target_shape=(4, )))
#    models.append(model_cat_05)
#
#    model_cat_06 = Sequential()
#    model_cat_06.add(Embedding(4, 2, input_length=1))
#    model_cat_06.add(Reshape(target_shape=(2, )))
#    models.append(model_cat_06)
#
#    model_cat_07 = Sequential()
#    model_cat_07.add(Embedding(60, 30, input_length=1))
#    model_cat_07.add(Reshape(target_shape=(30, )))
#    models.append(model_cat_07)
#
#    model_cat_08 = Sequential()
#    model_cat_08.add(Embedding(14, 7, input_length=1))
#    model_cat_08.add(Reshape(target_shape=(7, )))
#    models.append(model_cat_08)
#
#    model_cat_09 = Sequential()
#    model_cat_09.add(Embedding(3177, 50, input_length=1))
#    model_cat_09.add(Reshape(target_shape=(50, )))
#    models.append(model_cat_09)
#
#    model_cat_10 = Sequential()
#    model_cat_10.add(Embedding(874, 50, input_length=1))
#    model_cat_10.add(Reshape(target_shape=(50, )))
#    models.append(model_cat_10)
#
#    model_cat_11 = Sequential()
#    model_cat_11.add(Embedding(51, 26, input_length=1))
#    model_cat_11.add(Reshape(target_shape=(26, )))
#    models.append(model_cat_11)
#
#    model_cat_12 = Sequential()
#    model_cat_12.add(Embedding(606, 50, input_length=1))
#    model_cat_12.add(Reshape(target_shape=(50, )))
#    models.append(model_cat_12)
#
#    model_cat_13 = Sequential()
#    model_cat_13.add(Embedding(3, 2, input_length=1))
#    model_cat_13.add(Reshape(target_shape=(2, )))
#    models.append(model_cat_13)
#
#    model_cat_14 = Sequential()
#    model_cat_14.add(Embedding(59, 30, input_length=1))
#    model_cat_14.add(Reshape(target_shape=(30, )))
#    models.append(model_cat_14)
#
#    model_cat_15 = Sequential()
#    model_cat_15.add(Embedding(59, 30, input_length=1))
#    model_cat_15.add(Reshape(target_shape=(30, )))
#    models.append(model_cat_15)
#
#    model_cat_16 = Sequential()
#    model_cat_16.add(Embedding(4, 2, input_length=1))
#    model_cat_16.add(Reshape(target_shape=(2, )))
#    models.append(model_cat_16)
#
#    model_num = Sequential()
#    model_num.add(Dense(1024, input_dim=X_train[:, num_idx].shape[1]))
#    models.append(model_num)
#
#    model = Sequential()
#    # model.add(Embedding())
#    model.add(Merge(models, mode="concat"))
#    model.add(Dense(len(X_train), input_dim=X_train.shape[1], activation="relu"))
#    model.add(Dropout(0.3))
#    model.add(Dense(15, activation="relu"))
#    model.add(Dense(1, activation="sigmoid"))
#
#    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])
#    
#    return model


def NN_512(X_train):
    model = Sequential()
#    model.add(Dense(X_train.shape[0], input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(512, input_dim=X_train.shape[1], activation="relu", kernel_initializer='random_uniform')) #, kernel_initializer='random_uniform')
    model.add(Dropout(0.3))
    model.add(Dense(15, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
    return model

def NN_15(X_train):
    model = Sequential()
#    model.add(Dense(X_train.shape[0], input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(512, input_dim=X_train.shape[1], activation="relu", kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(15, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
    return model


def RF():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier()


def LR():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight="balanced")


def SVM():
    from sklearn import svm
    return svm.SVC(class_weight="balanced", cache_size=6000)

def linSVM():
    from sklearn import svm
    return svm.LinearSVC(class_weight="balanced")

    
