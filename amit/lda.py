'''
Amit Kumar
Roll No: 22CSM1R02
MTech-CSE, Sem01
DSF Assignment 1
'''
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from . import utils
from sklearn.model_selection import cross_validate

with open('./input.txt') as f:
    lines = f.readlines()

lda = LinearDiscriminantAnalysis()
lr_outputs = []
k = 5 # for 5 cross validations

def LDA():
    for i in range(len(lines)):
        data = utils.load_data(lines[i])
        data = utils.clean_data(data)
        x, y = utils.get_x_y(data)
        # print(x.shape, y.shape)
        output = utils.cross_validation(model=lda, _X=x, _y=y, _cv=k)
        lr_outputs.append(output)
    
    return lr_outputs
