from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import os
import pickle

F_final1 = {}
current_dic1 = {}
para_dict1 = {}
time1 = {}
path = '/Users/amrit/GITHUB/MAHAKIL_imbalance/dump/'
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        a = os.path.join(root, name)
        with open(a, 'rb') as handle:
            F_final = pickle.load(handle)
            F_final1 = dict(F_final1.items() + F_final.items())
print(F_final1)
