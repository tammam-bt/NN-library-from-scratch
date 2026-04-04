import numpy as np
from ..core.base import BaseEstimator
from ..core.losses import MSE

class LinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.weights = None
        self.bias = 0
    
    def fit(self, x, y, n_iterations = 1000):
        n_features = x.shape[1]
        self.weights = np.zeros(n_features)
        cost_iterations = []
        for i in range(n_iterations):
            dc_dw,dc_db = self.compute_gradient(x,y)
            self.weights -= self.alpha * dc_dw
            self.bias -= self.alpha * dc_db
            print(f"Iteration {i}: cost = {self.cost_function(x,y)}")
            cost_iterations.append(self.cost_function(x,y))
        return cost_iterations    
        
    def cost_function(self, x , y):
        return MSE(np.dot(x, self.weights) + self.bias, y)
    
    def compute_gradient(self, x, y):
        m,n = x.shape
        error = np.dot(x,self.weights) + self.bias - y
        dc_dw = np.array([np.dot(error ,x[:,j]) / m for j in range(n)])
        dc_db = np.sum(error) / m
        return dc_dw,dc_db
    
    def predict(self, x):
        return np.dot(x, self.weights) + self.bias 
        