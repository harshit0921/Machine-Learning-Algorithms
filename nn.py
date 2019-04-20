#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 00:50:31 2019

@author: shivamodeka
"""

from TransformData import transform
import pandas as pd
import torch
import torch.nn.functional as F
from torch import autograd, optim, nn
#from scipy.io import loadmat
import torch.utils.data

max_epochs = 1000


class NeuralNetwork(nn.Module):
    def __init__(self, n, Sl, activation_function):
        super().__init__()
        self.n = n
        self.layers = []
        self.functions = {'relu' : F.relu, 'identity' : self.identity, 
                          'tanh' : torch.tanh, 'sigmoid' : torch.sigmoid}
        self.fc1 = nn.Linear(in_features = Sl[0], out_features = Sl[1])
        self.fc2 = nn.Linear(in_features = Sl[1], out_features = Sl[2])
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)
        if (n == 4):
            self.fc3 = nn.Linear(in_features = Sl[2], out_features = Sl[3])
            self.layers.append(self.fc3)
        self.act_func = self.functions[activation_function]
        
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x=self.act_func(x)
        x = F.softmax(x, dim = 1)
        return x
        
    def identity(self, x):
        return x
    
    def train(self, X_train, y_train):
        X_input = autograd.Variable(X_train)
        y_target = autograd.Variable(y_train.long())
        opt = optim.SGD(params = self.parameters(), lr = 0.01)
        
        for epoch in range(max_epochs):
            self.zero_grad()
            output = self(X_input)
            predict = output.max(1)[1]
            loss = F.cross_entropy(output, y_target)
            loss.backward()
            opt.step()
        c = 1    
        for layer in self.layers:
            print('Layer = {}\nWeights = {}\nBias = {}'.format(c,layer.weight,layer.bias))
            c+=1
        print('Last layer activations = {}'.format(output))
        FP = (predict - y_target).nonzero().shape[0]
        print("Training Classification Error: {}".format(FP/y_target.shape[0]))
        
        if (self.n == 3):
            return self.fc1.weight, self.fc2.weight, self.fc1.bias, self.fc2.bias, output
        else:
            return self.fc1.weight, self.fc2.weight, self.fc3.weight,
            self.fc1.bias, self.fc2.bias, self.fc3.bias, output
        
    
    def predict(self, X_test, y_test):
        X_input = autograd.Variable(X_test)
        y_target = autograd.Variable(y_test.long())
        
        with torch.no_grad():
            y_output = self(X_input)
        
        _, prediction = y_output.max(1)
        FP = (prediction - y_target).nonzero().shape[0]
        print("Test Classification Error: {}\n".format(FP/(y_target.shape[0])))
        
        
def RunNN(n, Sl, X_train, y_train, X_test, y_test, activation_func):
    network = NeuralNetwork(n, Sl = Sl, activation_function = activation_func)
    network.train(X_train.transpose(0,1), y_train)
    network.predict(X_test.transpose(0,1), y_test)
    
    
def main():
    l = [i for i in range(30000)]
    l.extend(('Age','Male','Female','Label'))
    data = pd.read_csv('/Users/shivamodeka/Desktop/Machine-Learning-Algorithms/batches/batch0.csv', names = l)
    x,y = transform(data)
    
    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)
    
    act_func = 'sigmoid'
    n = x.shape[1]
    train_range = int(n * 0.8)
    x_train = x[0:train_range, :]
    y_train = y[0:train_range, :]
    x_test = x[train_range:n, :]
    y_test = y[train_range:n, :]
    Sl = [n, 50, 10, 2]
    RunNN(4, Sl, x_train, y_train, x_test, y_test, act_func)
        
main()