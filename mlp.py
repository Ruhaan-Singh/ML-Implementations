import numpy as np
import helper
import activations
import optimizers

class MLP:
    def __init__(self, layers:list):
        for i in range(len(layers)):
            print(layers[0])
            self.layers[i] = self.hidden(self, *(layers[i]))

    class hidden:
        def __init__(self, input_size, output_size, activation, weights=None, bias=None):
            if activation == "relu":
                self.weights = np.squeeze(helper.kaiming_init(input_size,output_size))
                self.bias = 0
            else:
                self.weights = np.squeeze(helper.xavier_init(input_size,output_size))
                self.bias = 0
