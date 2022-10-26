import numpy as np


class ActivationsLoss:
    def __init__(self, activation_function):
        self.activation_list = activation_function

    def check(self, position, data):
        # check activation function for layer and run forward and derivative
        if self.activation_list[position] == 'Sig':
            return self.sigmoid(data)

        elif self.activation_list[position] == 'Tanh':
            return self.tanh(data)

        elif self.activation_list[position] == 'Relu':
            return self.Relu(data)

        else:
            raise Exception("Provide valid activation function")

    def check_loss(self, position, data, target):
        # check loss function for layer and run forward and derivative

        if self.activation_list[position] == 'BCE':
            return self.binary_cross_entropy(target, data)

        # elif self.activation_list[position] == 'Tanh':
        #     print("letter is Tanh")

        # elif self.activation_list[position] == 'Relu':
        #     print("fruit is Relu")

        # elif self.activation_list[position] == 'SM':
        #     return self.softmax(data)

        else:
            raise Exception("Provide valid loss function")

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

    def tanh(self, data):
        # reference activation formulas
        tanh_forward = np.tanh(data)
        tanh_backward = 1 - np.square(tanh_forward)
        return tanh_forward, tanh_backward

    def Relu(self, data):
        # reference activation formulas
        Relu_forward = np.where(data >= 0, data, 0)
        Relu_backward = np.where(Relu_forward >= 0, 1, 0)
        return Relu_forward, Relu_backward

    def binary_cross_entropy(self, yreal, ypred):
        # reference
        pos = len(ypred)-1
        ypred = ypred[pos]
        ypred = np.clip(ypred, 1e-15, 1-1e-15)
        logprobs = -1*(yreal*np.log(ypred) + (1-yreal)*np.log(1-ypred))
        # # loss = -1.*yreal*np.log(ypred)
        # # loss = yreal-ypred
        # print(loss)
        # loss = np.mean(loss)
        m = yreal.shape[0]
        # logprobs = np.dot(yreal, np.log(ypred).T) + \
        #     np.dot((1-yreal), np.log((1-ypred)).T)
        loss = np.mean(logprobs/m)
        # makes sure loss is the dimension we expect. E.g., turns [[17]] into 17
        # loss = float(np.squeeze(loss))
        # assert (isinstance(loss, float))
        return loss
