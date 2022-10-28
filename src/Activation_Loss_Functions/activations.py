# References
# Derivative in Sigmoid Function, (Line 39-40) Available: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
# Derivative in Tanh Function, (Line 47) Available: https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
# Relu Function - Forward and Derivative, (Line 53-54) https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
# LogProbs in Binary Cross Entropy Function, (line65) https://pgg1610.github.io/blog_fastpages/python/pytorch/machine-learning/2021/02/05/NN_classification_from_scratch.html
# Clipping of Ypred, (line 65) https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/loss_functions.py
import numpy as np


class ActivationsLoss:
    def __init__(self, activation_function):
        # initialize the class with the activation list
        self.activation_list = activation_function

    def check(self, position, data):
        # check activation list with positionin the list being the particular layer and run forward and derivative
        if self.activation_list[position] == 'Sig':
            return self.sigmoid(data)

        elif self.activation_list[position] == 'Tanh':
            return self.tanh(data)

        elif self.activation_list[position] == 'Relu':
            return self.Relu(data)

        else:
            raise Exception("Provide valid activation function")

    def check_loss(self, position, data, target):
        # check loss function for layer and run loss

        if self.activation_list[position] == 'BCE':
            return self.binary_cross_entropy(target, data)

        else:
            raise Exception("Provide valid loss function")

    def sigmoid(self, data):
        # Sigmoid activation function
        # *Available: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
        sig_forward = 1/(1+np.exp(-data))  # forward function
        sig_backward = sig_forward*(1-sig_forward)  # derivative
        return sig_forward, sig_backward

    def tanh(self, data):
        # Tanh activation function
        tanh_forward = np.tanh(data)  # forward function
        # *Available: https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
        tanh_backward = 1 - np.square(tanh_forward)  # derivative
        return tanh_forward, tanh_backward

    def Relu(self, data):
        # Relu activation function
        # *Available : https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
        Relu_forward = np.where(data >= 0, data, 0)  # forward function
        Relu_backward = np.where(Relu_forward >= 0, 1, 0)  # derivative
        return Relu_forward, Relu_backward

    def binary_cross_entropy(self, yreal, ypred):
        # cross entropy calculation function
        pos = len(ypred)-1
        ypred = ypred[pos]
        # clip to avoid numpy exp overflow
        # *Available :https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/loss_functions.py
        ypred = np.clip(ypred, 1e-15, 1-1e-15)
        # *Availabe :https://pgg1610.github.io/blog_fastpages/python/pytorch/machine-learning/2021/02/05/NN_classification_from_scratch.html
        logprobs = -1*(yreal*np.log(ypred) + (1-yreal)*np.log(1-ypred))
        m = yreal.shape[0]
        loss = np.mean(logprobs/m)
        return loss
