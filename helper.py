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
    y = np.random.standard_normal(X_shape[0])

    return X,y


def kaiming_init(size:tuple):
   matrix = np.random.normal(0, 2/size[0], size) #[0] is for input dim

   return matrix


def Dataset(X_input, y_input, batch_size=1,):
    length = X_input.shape[0]

    if length % batch_size != 0:  #remove few examples fix later
       X_input = X_input[:(length//batch_size) * batch_size]
       y_input = y_input[:(length//batch_size) * batch_size]
       length = X_input.shape[0]

    reshaped = (length//batch_size,batch_size)
    new_shape = reshaped + X_input.shape[1:]

    X_input = np.reshape(X_input,new_shape)
    y_input = np.reshape(y_input,reshaped)

    return X_input,y_input
