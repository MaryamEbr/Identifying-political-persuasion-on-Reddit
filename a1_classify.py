#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from scipy import stats

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

import random
random.seed(19)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C)/np.sum(C)

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.divide(np.diagonal(C), np.sum(C, axis=1))

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.divide(np.diagonal(C), np.sum(C, axis=0))


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''

    # classifiers, fit
    clf1 = SGDClassifier().fit(X_train, y_train)
    clf2 = GaussianNB().fit(X_train, y_train)
    clf3 = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_train, y_train)
    clf4 = MLPClassifier(alpha=0.05).fit(X_train, y_train)
    clf5 = AdaBoostClassifier().fit(X_train, y_train)


    # prediction, confusion matrix
    y_pred1 = clf1.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred1)
    y_pred2 = clf2.predict(X_test)
    cm2 = confusion_matrix(y_test, y_pred2)
    y_pred3 = clf3.predict(X_test)
    cm3 = confusion_matrix(y_test, y_pred3)
    y_pred4 = clf4.predict(X_test)
    cm4 = confusion_matrix(y_test, y_pred4)
    y_pred5 = clf5.predict(X_test)
    cm5 = confusion_matrix(y_test, y_pred5)

    name_list = ['SGDClassifier', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier', 'AdaBoostClassifier']
    cm_list = [cm1, cm2, cm3, cm4, cm5]
    acc_list = [accuracy(cm1), accuracy(cm2), accuracy(cm3), accuracy(cm4), accuracy(cm5)]
    recall_list = [recall(cm1), recall(cm3), recall(cm3), recall(cm4), recall(cm5)]
    precision_list = [precision(cm1), precision(cm2), precision(cm3), precision(cm4), precision(cm5)]



    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for classifier_name, acc, recalli, precisioni, conf_matrix in zip(name_list, acc_list, recall_list, precision_list, cm_list):
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recalli]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precisioni]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    ##  best classifier based on highest accuracy
    iBest = np.argmax(acc_list)
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    size_list = [1000, 5000, 10000, 15000, 20000]
    acc_list = []

    for s in size_list:
        X_train_sample, y_train_sample = zip(*random.sample(list(zip(X_train, y_train)), s))

        # setting the return values
        if (s == 1000):
            X_1k, y_1k = X_train_sample, y_train_sample

        if (iBest == 0):
            clf = SGDClassifier().fit(X_train_sample, y_train_sample)
        if (iBest == 1):
            clf = GaussianNB().fit(X_train_sample, y_train_sample)
        if (iBest == 2):
            clf = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_train_sample, y_train_sample)
        if (iBest == 3):
            clf = MLPClassifier(alpha=0.05).fit(X_train_sample, y_train_sample)
        if (iBest == 4):
            clf = AdaBoostClassifier().fit(X_train_sample, y_train_sample)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc_list.append(accuracy(cm))


    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for num_train, acc in zip(size_list, acc_list):
            outf.write(f'{num_train}: {acc:.4f}\n')

    return (X_1k, y_1k)

def class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    X_32k = X_train
    y_32k = y_train

    # part 1
    sorted_features_list = []
    sorted_pps_list = []
    k_list = [5, 50]
    for ki in k_list:
        selector = SelectKBest(f_classif, k=ki)
        X_new = selector.fit_transform(X_32k, y_32k)
        pp = selector.pvalues_

        sorted_features_list.append(selector.get_support(indices=True))
        sorted_pps_list.append([pp[i] for i in sorted_features_list[-1]])



    # part 2
    X_1k_top5_feats = [[a[i] for i in sorted_features_list[0]] for a in X_1k]
    X_32k_top5_feats = [[a[i] for i in sorted_features_list[0]] for a in X_32k]
    X_test_top5_feats = [[a[i] for i in sorted_features_list[0]] for a in X_test]

    if (iBest == 0):
        clf_1k = SGDClassifier().fit(X_1k_top5_feats, y_1k)
        clf_32k = SGDClassifier().fit(X_32k_top5_feats, y_32k)
    if (iBest == 1):
        clf_1k = GaussianNB().fit(X_1k_top5_feats, y_1k)
        clf_32k = GaussianNB().fit(X_32k_top5_feats, y_32k)
    if (iBest == 2):
        clf_1k = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_1k_top5_feats, y_1k)
        clf_32k = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_32k_top5_feats, y_32k)
    if (iBest == 3):
        clf_1k = MLPClassifier(alpha=0.05).fit(X_1k_top5_feats, y_1k)
        clf_32k = MLPClassifier(alpha=0.05).fit(X_32k_top5_feats, y_32k)
    if (iBest == 4):
        clf_1k = AdaBoostClassifier().fit(X_1k_top5_feats, y_1k)
        clf_32k = AdaBoostClassifier().fit(X_32k_top5_feats, y_32k)

    y_pred = clf_1k.predict(X_test_top5_feats)
    accuracy_1k = accuracy(confusion_matrix(y_test, y_pred))

    y_pred = clf_32k.predict(X_test_top5_feats)
    accuracy_full = accuracy(confusion_matrix(y_test, y_pred))


    # part 3
    selector = SelectKBest(f_classif, k=5)
    selector.fit_transform(X_32k, y_32k)
    top5_32k = selector.get_support(indices=True)

    selector = SelectKBest(f_classif, k=5)
    selector.fit_transform(X_1k, y_1k)
    top5_1k = selector.get_support(indices=True)

    feature_intersection = list(set(top5_32k) & set(top5_1k))


    # paer 4
    top_5 = top5_32k

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        for k_feat, p_values in zip(k_list, sorted_pps_list):
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')



def class34(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''

    # ???? don't change the header and combine the two
    kf = KFold(n_splits=5, shuffle=True)
    acc_list = []
    p_values = []
    fold = 0

    # combining data again
    X = np.array(X_train.tolist() + X_test.tolist())
    y = np.array(y_train.tolist() + y_test.tolist())


    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # classifiers, fit
        clf1 = SGDClassifier().fit(X_train, y_train)
        clf2 = GaussianNB().fit(X_train, y_train)
        clf3 = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_train, y_train)
        clf4 = MLPClassifier(alpha=0.05).fit(X_train, y_train)
        clf5 = AdaBoostClassifier().fit(X_train, y_train)

        # predict, confusion matrix
        acc1 = accuracy(confusion_matrix(y_test, clf1.predict(X_test)))
        acc2 = accuracy(confusion_matrix(y_test, clf2.predict(X_test)))
        acc3 = accuracy(confusion_matrix(y_test, clf3.predict(X_test)))
        acc4 = accuracy(confusion_matrix(y_test, clf4.predict(X_test)))
        acc5 = accuracy(confusion_matrix(y_test, clf5.predict(X_test)))

        acc_list.append([acc1, acc2, acc3, acc4, acc5])



    for i in range(5):
        if(i != iBest):
            S = stats.ttest_rel(np.array(acc_list).T[i], np.array(acc_list).T[iBest])
            p_values.append(S.pvalue)

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        for kfold_accuracies in acc_list:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    data = np.load(args.input)['arr_0']
    data_x = data[:, 0:173]
    data_y = data[:, 173]

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, train_size=0.8, random_state=19)

    # TODO : complete each classification experiment, in sequence.
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

