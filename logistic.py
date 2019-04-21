import numpy as np
from scipy import optimize
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Logistic():
    def __init__(self):
        self.theta = np.zeros((self.Xtrain.shape[1], 1))
    
    def train(self, x_train, y_train):
        self.Xtrain = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis = 1)
        self.ytrain = y_train[:, np.newaxis]
        self.lamb = self.k_fold_cross_validation(self.Xtrain, self.ytrain, 10)
        self.theta = self.fit(self.Xtrain, self.ytrain, self.theta)
        return self.theta
        
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
                       (1 - y) * np.log(1 - self.probability(theta, x))) + self.lamb * np.linalg.norm(theta, ord = 2)
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
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
        theta = self.theta[:, np.newaxis]
        return (self.probability(theta, x) >= 0.5).astype(int).flatten()
    
    def k_fold_cross_validation(self, x, y, K):
        div = len(x)//K
        min_lambda_error = float("inf")
        min_lambda = 0
        
    
        for lam in np.arange(0.1, 2, 0.1):
            self.lamb = lam
            start_index = 0
            end_index = div
            error = 0
            epsilon = 0
            for i in range(K):
                
                #Prepare test data and training data for fold K
                Dx_tst = x[start_index : end_index, :]
                Dy_tst = y[start_index : end_index]
                Dx_trn = np.concatenate((x[: start_index, :], x[end_index : , :]), axis = 0)
                Dy_trn = np.concatenate((y[: start_index], y[end_index : ]), axis = 0)
                start_index += div
                end_index += div
                
                #Calculate theeta for current fold
                theeta = self.fit(Dx_trn, Dy_trn, np.zeros((self.Xtrain.shape[1], 1)))
                
                #Calculate error for this lambda value at current fold
                for i in range(len(Dx_tst)):
                    error += self.cost_function(theeta, Dx_tst, Dy_tst)
                
            epsilon = error/K
            
            #Get lambda and theeta for minimum error
            if(epsilon < min_lambda_error):
                min_lambda_error = epsilon
                min_lambda = lam
        return min_lambda
    
#def main():
#    table = pd.read_csv('breast_cancer_data.csv')
#    train_range = int(table.values.shape[0] * 0.8)
#    x_train = StandardScaler().fit_transform(table.values[0:train_range, :-1])
#    y_train = table.values[0:train_range, -1]
#    x_test = StandardScaler().fit_transform(table.values[train_range:table.values.shape[0], :-1] )
#    y_test = table.values[train_range:table.values.shape[0], -1]
#    logistic = Logistic(x_train, y_train)
#    print(logistic.parameters)
#    y_pred = logistic.predict(x_test)
#    print(y_pred)
#    print(y_test.flatten())
#    print(np.sum(np.abs(y_pred - y_test)))
#main()  