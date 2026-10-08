import numpy as np

def sigmoid(x: float):
    return (1/(1 + np.exp(x)))

def sigmoid(x: float):
    return np.maximum(0,x)

def tanh(x: float):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))