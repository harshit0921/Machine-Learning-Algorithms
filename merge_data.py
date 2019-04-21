#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:05:25 2019

@author: shivamodeka
"""

#from TransformData import transform
import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

def merge():

    script_location = os.path.dirname(__file__)
    
    l = [i for i in range(30000)]
    l.extend(('Age','Male','Female','Label'))
    features = [i for i in range(30000)]
    features.extend(('Age','Male','Female'))
    
    data = pd.read_csv(os.path.join(script_location, 'batches/batch' + str(0) + '.csv'), names =l)
    x = data.loc[:, features].values
    y = data.loc[:,['Label']].values
    
    for i in range(1,100):
        data = pd.read_csv(os.path.join(script_location, 'batches/batch' + str(i) + '.csv'), names =l)
        x1 = data.loc[:, features].values
        y1 = data.loc[:,['Label']].values
        x= np.append(x,x1, axis =0)
        y= np.append(y,y1, axis =0)
        
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=1/5.0, random_state=0)
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    scale = StandardScaler()
    
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)
    
    pca = PCA(.99)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype = torch.float32) 
    y_train = torch.tensor(y_train, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32) 
    
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    merge()
