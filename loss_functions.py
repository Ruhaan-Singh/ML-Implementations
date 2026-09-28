import numpy as np

class MSE:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def value(y_true,y_pred):
        return np.sum((y_true - y_pred)**2)

    def backwards(self, weights, bias, y_true, y_pred):
        self.weight += 2 * self.learning_rate * (y_true-y_pred) * self.weight
        self.bias += 2 * self.learning_rate * (y_true - y_pred)

        return self.weight,self.bias