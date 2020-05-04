# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:05:14 2018

@author: Shin-PC
"""
import numpy as np

x_data = np.loadtxt('pre_score.txt',delimiter=',', dtype=np.float64)
ex_score = np.loadtxt('ex_score.txt')

y_after = []
for i in range(len(x_data)):
    if x_data[i] > ex_score[i]:
        y_after.append(1)
    else:
        y_after.append(0)
y_data = np.array(y_after)
print(x_data)
print(y_data)



