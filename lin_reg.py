import numpy as np
import helper

class LinearRegression:
        def __init__(self,weights,bias):
                self.weights = weights
                self.bias = bias

        
        def train(self, X,y, epochs):
                for i in range(len(epochs)):
                        y_pred = self.weights @ X + self.bias






        