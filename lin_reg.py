import numpy as np
import helper
import loss_functions
import optimizers

class LinearRegression:
        def __init__(self, input_size, weights=None,bias=None):
                self.weights = helper.kaiming_init((input_size,))
                self.bias = 0

        def forward(self, X):
                y_pred = X @ self.weights + self.bias
                return y_pred

        def backwards(self, X_batch, y_batch, y_pred,loss_function,optimizer):
                w_grad,b_grad = loss_function.gradient(X_batch,y_batch,y_pred)

                self.weights = optimizer.update(self.weights,w_grad)
                self.bias = optimizer.update(self.bias,b_grad)

                return self.weights,self.bias

        def train(self, X_train,y_train, epochs,loss_function,optimizer):
                loss_values = [0] * epochs
                for i in range(epochs):
                        loss = 0
                        order = np.random.permutation(X_train.shape[0])
                        for j in order:
                                X_batch = X_train[j]
                                y_batch = y_train[j]

                                y_pred =  self.forward(X_batch)
                                loss += loss_function.value(y_batch,y_pred)

                                self.weights,self.bias =  self.backwards(X_batch, y_batch, y_pred,loss_function,optimizer)

                        loss /= X_train.shape[0]
                        loss_values[i] = loss

                return loss_values

                        







        