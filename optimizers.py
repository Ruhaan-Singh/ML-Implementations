import numpy as np
import helper
import loss_functions

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, param, gradient):
        param += self.learning_rate * gradient

        return param