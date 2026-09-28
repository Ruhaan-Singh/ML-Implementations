import numpy as np
import helper
import loss_functions

class LinearRegression:
        def __init__(self,weights,bias):
                self.weights = weights
                self.bias = bias

        def train(self, X_train,y_train, epochs,loss_function,return_loss=False):
                loss_values = [0] * epochs

                for i in range(len(epochs)):
                        loss = 0
                        for j in range(len(X_train.shape[0])):
                                X_batch = X_train[j]
                                y_batch = y_train[j]

                                y_pred =  X_batch @ self.weights + self.bias
                                loss += loss_function.value((y_batch,y_pred))
                                self.weights,self.bias = loss_function.backward(self.weights, self.bias, y_batch, y_pred)

                        if return_loss:
                                loss_values[i]  = loss

                if return_loss:
                        return loss_values

        def predict(self, X_test):
                y_pred = X_test @ self.weights + self.bias
                
                return y_pred
                        







        