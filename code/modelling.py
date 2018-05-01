#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:45:36 2018

@author: spaceape
"""
import pandas as pd

# return vectors with sorted scores for each feature for given statistics
def best_features(X, y, features, statistics="chi2"):
    from sklearn.feature_selection import SelectKBest, f_classif, chi2
    import pandas as pd
    if statistics=="chi2":
        selector = SelectKBest(chi2)
        selector.fit(X, y)
        mask = selector.get_support()
        columns = []
        for col in range(X.shape[1]):
            if mask[col] == True:
                columns.append(features[col])
        score = pd.DataFrame(selector.scores_, columns=["score"], index=features)
        score = score.sort_values(by=["score"], ascending=False) 
    elif statistics=="anova":
        selector = SelectKBest(f_classif)
        selector.fit(X, y)
        mask = selector.get_support()
        columns = []
        for col in range(X.shape[1]):
            if mask[col] == True:
                columns.append(features[col])
        score = pd.DataFrame(selector.scores_, columns=["score"], index=features)
        score = score.sort_values(by=["score"], ascending=False)    
    else:
        print("not implemented statistics")  
    return score 


def feature_selection(X, y, statistics="chi2", k=10):
    from sklearn.feature_selection import SelectKBest, chi2
    if statistics=="chi2":
        if k <= X.shape[1]:    
            selector = SelectKBest(chi2, k=k)
            X = selector.fit_transform(X, y)
        else:
            print("number of inputed features is lower than given parameter k")
    else:
        print("currently is implement only chi2 test")
    return X


# choose best parameters
def choose_best(scores1, scores2, features, k=10):
    ranks = scores1.rank()
    ranks = ranks.append(scores2.rank())
    ranks = ranks.groupby(ranks.index).sum().sort_values(by=["score"], ascending=False)
#    for i in range(k):
#        best.append(ranks.index[i])
    tmp = []
    for i in ranks[0:k].index:
        for j in features.index:
            if i == features[j]:
                tmp.append(j)
    best = pd.Series(data=tmp, index=ranks[0:k].index)
    return best



def find_hyperparams(classifier, parameters, X, y, search_method, n_iter=1, refit=True, cv=10):
    from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
    
    if search_method == "gridsearch":
        clf = GridSearchCV(classifier, parameters, refit=refit, cv=cv) #, scoring="f1"
        clf.fit(X, y)
        return clf, clf.best_params_
    else:
        clf = RandomizedSearchCV(classifier, parameters, n_iter=n_iter, refit=refit, cv=cv) #, scoring="f1"
        clf.fit(X, y)
        return clf, clf.best_params_
    

def evaluation(classifier, X_test, y_test, y_pred, ROC=False):
    from sklearn import metrics
    import matplotlib as plt
    print ("accuracy score: ", metrics.accuracy_score(y_test, y_pred))
    print ("confusion matrix: ", "\n", metrics.confusion_matrix(y_test, y_pred))
    print ("precision score: ",  metrics.precision_score(y_test, y_pred))
    print ("recall score: ",  metrics.recall_score(y_test, y_pred))
    print ("F1 score: ",  metrics.f1_score(y_test, y_pred))
    print ("AUC: ", metrics.roc_auc_score(y_test, y_pred))

    score = {"accuracy score": metrics.accuracy_score(y_test, y_pred),\
    "confusion matrix": metrics.confusion_matrix(y_test, y_pred), \
    "precision score": metrics.precision_score(y_test, y_pred), \
    "recall score": metrics.recall_score(y_test, y_pred), \
    "F1 score": metrics.f1_score(y_test, y_pred), \
    "AUC" : metrics.roc_auc_score(y_test, y_pred)} 
    return score
    
    
    if ROC == True:
        fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test, y_pred) #classifier.predict_proba(X_test)[:,1]
            
        plt.figure(figsize=(10,6))
        plt.plot(fpr_lr, tpr_lr, color='darkorange', label='Classifier (area = %0.2f)' % metrics.auc(fpr_lr, tpr_lr))
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("ROC.eps")
        plt.show()
    