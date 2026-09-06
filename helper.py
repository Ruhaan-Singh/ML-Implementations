import numpy as np

def normalize(array_in, dim=(1)):

    min_values = np.min(array_in, axis=dim, keepdims=True)
    max_values = np.max(array_in, axis=dim, keepdims=True)

    array_out = (array_in - min_values)/(max_values - min_values)

    return array_out


def standardize(array_in, dim=(1), episilon=1e-7):
    means = np.mean(array_in, axis=dim, keepdims=True)
    std_dev = np.std(array_in, axis=dim, keepdims=True)

    array_out = (array_in - means)/(std_dev + episilon)

    return array_out


