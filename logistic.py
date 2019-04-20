import numpy as np
from scipy import optimize
import torch

class Logistic():
    def __init__(self, Xtrain, ytrain):
        self.Xtrain = torch.cat((torch.ones(Xtrain.shape[0], 1), 
                                 torch.tensor(Xtrain, dtype = torch.float)), dim = 1).numpy()
        self.ytrain = torch.tensor(ytrain[:, np.newaxis]).numpy()
        self.theta = torch.zeros(self.Xtrain.shape[1], 1).numpy()
        self.parameters = self.fit(self.Xtrain, self.ytrain, self.theta)
        
    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)
    
    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        return self.sigmoid(self.net_input(theta, x))
    
    def cost_function(self, theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(y * np.log(self.probability(theta, x)) + 
                       (1 - y) * np.log(1 - self.probability(theta, x)))
        return total_cost
    
    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)
    
    def flatten(t):
        t = t.reshape(1, -1)
        t = t.squeeze()
        return t

    def fit(self, x, y, theta):
        opt_weights = optimize.fmin_tnc(func=self.cost_function, x0=theta, 
                                        fprime=self.gradient, args=(x, y.flatten()))
        return opt_weights[0]
    
    def predict(self, x):
        theta = self.parameters[:, np.newaxis]
        return (self.probability(theta, x) >= 0.5).astype(int).flatten()
    
#def main():
#    table = loadmat('dataset3.mat')
#    x_training = table['X_trn']
#    y_tr = table['Y_trn']
#    x_test = table['X_tst'] 
#    y_tst = table['Y_tst']
#    logistic = Logistic(x_training, y_tr)
#    print(logistic.parameters)
#    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis = 1)
#    y_pred = logistic.predict(x_test)
#    print(y_pred)
#    print(y_tst)
#
#main()  