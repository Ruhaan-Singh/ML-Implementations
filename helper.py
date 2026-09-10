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

def synth_data(X_shape:tuple):
    X = np.random.standard_normal(X_shape)
    y = np.random.standard_normal(1)

    return X,y

def Dataset(X_input, y_input, batch_size=1,):
    X_batched = 1

    return


def kaiming_init(size:tuple):
   matrix = np.random.normal(0, 2/size[0], size) #[0] is for input dim

   return matrix
