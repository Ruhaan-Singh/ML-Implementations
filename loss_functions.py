import numpy as np

class MSE:
    def __init__(self,learning_rate=0.001):
        self.learning_rate = learning_rate

    def value(self,y_true,y_pred):
        return np.sum((y_true - y_pred)**2)/(y_true.shape[0])

    def backwards(self, weights, bias, X, y_true, y_pred):
        weights += 2 * self.learning_rate * (X.T @ (y_true-y_pred))/(y_true.shape[0])
        bias += 2 * self.learning_rate * np.sum(y_true - y_pred)/(y_true.shape[0])

        return weights,bias