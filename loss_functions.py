import numpy as np

class MSE:
    def value(self,y_true,y_pred):
        return np.sum((y_true - y_pred)**2)/(y_true.shape[0])

    def gradient(self,X, y_true, y_pred):
        weight_grad = 2 * (X.T @ (y_true-y_pred))/(y_true.shape[0])
        bias_grad = 2  * np.sum(y_true - y_pred)/(y_true.shape[0])

        return weight_grad,bias_grad

class LogLoss():
    def value(self,y_true,y_pred):
        return np.sum(y_true.T @ np.log(y_pred) + (1-y_true).T @ np.log(1-y_pred))/(-1*y_true.shape[0])

    def gradient(self,X,y_true,y_pred):
        pass