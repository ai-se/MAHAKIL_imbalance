from __future__ import print_function, division

__author__ = 'amritanshu.agrawal'

import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def DT(train_data,train_label,test_data):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(train_data, train_label)
    return model.predict(test_data)

def KNN(train_data,train_label,test_data):
    model = neighbors.KNeighborsClassifier(n_neighbors=8)
    model.fit(train_data, train_label)
    return model.predict(test_data)

def LR(train_data,train_label,test_data):
    model = LogisticRegression()
    model.fit(train_data, train_label)
    return model.predict(test_data)

def SVM(train_data,train_label,test_data):
    model = SVC(kernel='linear')
    model.fit(train_data, train_label)
    return model.predict(test_data)

def NB(train_data,train_label,test_data):
    model = GaussianNB()
    model.fit(train_data, train_label)
    return model.predict(test_data)

def RF(train_data,train_label,test_data):
    model = RandomForestClassifier(criterion='entropy')
    model.fit(train_data, train_label)
    return model.predict(test_data)