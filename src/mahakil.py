from __future__ import print_function, division

__author__ = 'amritanshu.agrawal'

import sys

sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
from Learners import *
from ABCD import ABCD
from sklearn.metrics import roc_curve, auc
import pickle
from demos import cmd

learners=[NB, KNN, DT,LR, SVM, RF]
#files = ['tomcat','synapse', 'camel', 'ant', 'arc', 'ivy', 'velocity', 'redaktor', 'jedit']

def cut_position(pos, neg, percentage=0):
    return int(len(pos) * percentage / 100), int(len(neg) * percentage / 100)

def divide_train_test(pos, neg, cut_pos, cut_neg):
    train_pos, train_neg = list(pos)[:cut_pos], list(neg)[:cut_neg]
    data_test, test_label = list(pos)[cut_pos:] + list(neg)[cut_neg:], [1] * (len(pos) - cut_pos) + [0] * (
        len(neg) - cut_neg)
    return np.array(train_pos), np.array(train_neg), np.array(data_test), test_label

def update_parents(parents):
    temp=[]
    for i in parents:
        instance = [sum(e) for e in zip(*i)]
        temp.append([i[0],instance])
        temp.append([instance,i[1]])
    return temp

def _test(res=''):
    files=[]
    files.append(res)
    final={}
    for f in files:
        df=pd.read_csv('../data/'+f+'.csv')
        df_def=df[df['bug']==1]
        df_non=df[df['bug']==0]
        pos=df_def.drop(['bug'],axis=1)
        neg = df_non.drop(['bug'], axis=1)
        pos=np.array(pos.values.tolist())
        neg=np.array(neg.values.tolist())
        result = {}
        cut_pos, cut_neg = cut_position(pos, neg, percentage=80)
        for lea in learners:
            dic={}
            print(lea.__name__)
            #dic[lea.__name__]={}
            measures = ["Recall", "Precision", "Accuracy", "F_score", "False_alarm", "AUC"]
            for q in measures:
                dic[q]=[]
            for folds in range(15):
                pos_shuffle = range(0, len(pos))
                neg_shuffle = range(0, len(neg))
                train_pos, train_neg, data_test, test_label = divide_train_test(pos, neg, cut_pos, cut_neg)
                T=len(train_neg)-len(train_pos)
                V = np.cov(train_pos.T)
                #VI = np.linalg.inv(V)
                m = 10 ^ -6
                VI = np.linalg.inv(V + np.eye(V.shape[1]) * m)

                mah_dis_matrix=np.diag(np.sqrt(np.dot(np.dot(train_pos, VI), train_pos.T)))
                train=zip(list(train_pos),list(mah_dis_matrix))
                train=sorted(train, key=lambda x: x[1],reverse=True)
                train=[list(x) for x,y in train]
                mid=len(train)//2
                par_1, par_2=train[:mid],train[mid:]
                parents=zip(par_1,par_2)
                if T>0:
                    count=0
                    while count<=T:
                        parents=update_parents(parents)
                        count=len(parents)
                    temp=[]
                    for i in range(0,len(parents),2):
                        temp.append(parents[i][0])
                        temp.append(parents[i][1])
                        temp.append(parents[i+1][1])
                    train_pos=np.array(temp)
                    data_train = np.concatenate((train_pos, train_neg))
                    train_label = len(train_pos) * [1] + len(train_neg) * [0]
                else:
                    data_train=np.concatenate((train_pos,train_neg))
                    train_label=len(train_pos)*[1] + len(train_neg)*[0]
                prediction=lea(data_train,train_label,data_test)
                abcd = ABCD(before=test_label, after=prediction)
                uniques = list(set(test_label))
                stats = np.array([j.stats() for j in abcd()])
                if uniques[0] == 0:
                    target = 1
                else:
                    target = 0
                fpr, tpr, _ = roc_curve(test_label, prediction, pos_label=target)
                dic["AUC"].append(auc(fpr, tpr))
                dic["Recall"].append(stats[target][0])
                dic["Precision"].append(stats[target][3])
                dic["Accuracy"].append(stats[target][4])
                dic["F_score"].append(stats[target][5])
                dic["False_alarm"].append(stats[target][1])
            result[lea.__name__]=dic
        final[f]=result
    with open('../dump/'+res+'.pickle', 'wb') as handle:
        pickle.dump(final, handle)

if __name__ == '__main__':
    eval(cmd())