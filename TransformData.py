#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:02:26 2019

@author: shivamodeka
"""

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

def transform(data):
    
    features = [i for i in range(30000)]
    features.extend(('Age','Male','Female'))
    x = data.loc[:, features].values
    y = data.loc[:,['Label']].values
#    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components =100)
    pc= pca.fit_transform(x)
    
    return pc,y
