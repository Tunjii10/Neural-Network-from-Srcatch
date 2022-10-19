import numpy as np


class Activations:
    def __init__(self, activation_function, data):
        if activation_function == 'Sig':
            self.sigmoid(data)

        elif activation_function == 'Tanh':
            print("letter is Tanh")

        elif activation_function == 'Relu':
            print("fruit is Relu")

        else:
            raise Exception("Provide valid activation function")

    def sigmoid(self, data):
        sig = 1./(1.+np.exp(-data))
        return sig, sig*(1.-sig)
