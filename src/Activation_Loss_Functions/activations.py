import numpy as np


class ActivationsLoss:
    def __init__(self, activation_function):
        self.activation_list = activation_function

    def check(self, position, data):

        if self.activation_list[position] == 'Sig':
            return self.sigmoid(data)

        elif self.activation_list[position] == 'Tanh':
            print("letter is Tanh")

        elif self.activation_list[position] == 'Relu':
            print("fruit is Relu")

        elif self.activation_list[position] == 'SM':
            return self.softmax(data)

        else:
            raise Exception("Provide valid activation function")

    def sigmoid(self, data):
        ''' Reference ↓ ↓ ↓
            *    Title/Project Name: Multi-Layer-Neural-Network-from-scratch
            *    Author: bp249507
            *    Date: 2017
            *    Lines: 35-36,41-42
            *    Availability: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
        '''

        sig_forward = 1/(1+np.exp(-data))
        sig_backward = sig_forward*(1-sig_forward)
        return sig_forward, sig_backward

    def softmax(self, data):
        # reference
        t = np.exp(data)
        softmax = t/(np.sum(t, axis=1).reshape((len(t), 1)))
        return softmax, "no_back_propagate"
