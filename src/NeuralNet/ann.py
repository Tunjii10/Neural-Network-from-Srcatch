import os
import numpy as np
import pandas as pd
from Activation_Loss_Functions import activations


class Ann:
    def __init__(self, neurons_in_layers, activations_loss, learning_rate, loss_function, grad_descent):
        np.random.seed(0)  # numpy random seed

        # initialize params
        self.network = dict()
        self.layer_input_output_pairs = []
        self.m = None

        # eval string inputs => array, integer
        self.neurons_in_layers = eval(neurons_in_layers)
        self.activation_loss = eval(activations_loss)
        self.learning_rate = eval(learning_rate)
        self.loss_function = eval(loss_function)

        # initialize  activation class for loss function used
        self.init_loss_activate = activations.ActivationsLoss(
            self.loss_function)

        ''' Reference ↓ ↓ ↓
            *    Title/Project Name: Multi-Layer-Neural-Network-from-scratch
            *    Author: bp249507
            *    Date: 2017
            *    Lines: 21-23,28-31, 44-45, 50-51, 53-55,
            *    Availability: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
        '''
        # loop through neuron layers to get structure pair for weight dimensions
        for i in range(len(self.neurons_in_layers)-1):
            self.layer_input_output_pairs.append(
                (self.neurons_in_layers[i], self.neurons_in_layers[i+1]))

        # create  network by initializing weights and bias using structure pairs
        for x in range(len(self.layer_input_output_pairs)):
            Weight = np.random.randn(
                self.layer_input_output_pairs[x][0], self.layer_input_output_pairs[x][1])
            bias = np.zeros((1, self.layer_input_output_pairs[x][1]))
            self.network[x] = {'W': Weight, 'b': bias}
        self.network_length = len(self.network)
        print("network bias and weights \n >>>>>>>> \n {}".format(self.network))

    def forward_propagation(self, x):

        # run checks to see if input correlates with data
        if self.neurons_in_layers[0] != x.shape[1]:
            raise Exception(
                "Your specified inputs no doesnt match x dataa provided")
        elif len(self.activation_loss) != self.network_length:
            raise Exception(
                "Activation functions specified dont match layers provided")
        else:
            self.m = x.shape[0]  # number of samples
            a = {}
            da = {}

            activate = activations.ActivationsLoss(
                self.activation_loss)  # initialize  activation class

            # forward propagate input through network
            for y in range(self.network_length):
                if y == 0:
                    Wx = x.dot(self.network[y]['W'])
                    b = self.network[y]['b']
                else:
                    Wx = a[y-1].dot(self.network[y]['W'])
                    b = self.network[y]['b']
                z = Wx + b
                forw_a, back_a = activate.check(
                    y, z)  # network outputs forward activation and its derivative
                a[y] = forw_a
                if (self.network_length-1) != y:  # for last layer activation no derivative taken
                    da[y] = back_a
        return a, da, x  # outputs forward activation, derivative of activation and x input

    def loss(self, a, y):

        loss = self.init_loss_activate.check_loss(
            0, a, y)
        print("Loss  = {}".format(loss))  # print loss

    def backward_propagation(self, a, da, datax, y):
        dz = {}
        db = {}
        dw = {}
        # loop backwards through length 'a' == length of forward activations gotten in forward pass
        for x in range(len(a)-1, -1, -1):

            ''' Reference ↓ ↓ ↓
                *    Title/Project Name: Challenge 1 Bletchley Three Layer NNetwork
                *    Author: JANTINBERGEN
                *    Date: 2017
                *    Lines: 77-79,81-84, 86-89, 50-51, 53-55,
                *    Availability: https://www.kaggle.com/code/jantinbergen/challenge-1-bletchley-three-layer-nnetwork
            '''
            # backpropagate by finding dz,dw,db for each layer
            if x == len(a)-1:  # last layer dz = y'-y
                dz[x] = a[x] - y
                dw[x] = 1/self.m*(a[x-1].T).dot(dz[x])
                db[x] = 1/self.m*np.sum(dz[x], axis=0)
            elif x == 0:  # first layer
                print(self.network[x+1]['W'].T)
                print(da[x])
                dz[x] = np.multiply(
                    dz[x+1].dot(self.network[x+1]['W'].T), da[x])
                # a[0]==our original input i.e datax
                dw[x] = 1/self.m*np.dot(datax.T, dz[x])
                db[x] = 1/self.m*np.sum(dz[x], axis=0)
            else:
                dz[x] = np.multiply(
                    dz[x+1].dot(self.network[x+1]['W'].T), da[x])
                dw[x] = 1/self.m*np.dot(a[x-1].T, dz[x])
                db[x] = 1/self.m*np.sum(dz[x], axis=0)
        return dw, db

    def update_parameters(self, dw, db):
        # for each layer update weights and biases
        for layer in range(len(self.network)):

            ''' Reference ↓ ↓ ↓
                *    Title/Project Name: Multi-Layer-Neural-Network-from-scratch
                *    Author: bp249507
                *    Date: 2017
                *    Lines: 103-104
                *    Availability: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
            '''

            self.network[layer]['W'] -= self.learning_rate*dw[layer]
            self.network[layer]['b'] -= self.learning_rate*db[layer]
        return self.network

    def predict(self, x):
        # not done yet
        ''' Reference ↓ ↓ ↓
            *    Title/Project Name: Multi-Layer-Neural-Network-from-scratch
            *    Author: bp249507
            *    Date: 2017
            *    Lines: 103-104
            *    Availability: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
        '''

        a_, _, _ = self.forward_propagation(x)
        return a_[len(a_)-1].round()


def pipeline(neurons_in_layers, activations, learning_rate, epoch, loss_function, d):
    '''
        Step by step working of the neural network
        for each epooch = forward propagate -> back propagate -> update weight and biases
    '''
    myNeuralNet = Ann(neurons_in_layers, activations,
                      learning_rate, loss_function, d)
    # refence data import
    path = os.getcwd()
    data = pd.read_csv(path+"/data/wdbc.data", sep=",", header=None)
    x = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    for ind, val in enumerate(y):
        if val == 'M':
            y[ind] = 1
        else:
            y[ind] = 0

    print(x.head())
    print(y.head(20))
    # x = np.array([[-2.7810836, -2.5505370], [7.62753121, 6.7592622], [
    #              -1.4654893, -2.3621250], [-1.3880701, -1.8502203], [7.6455312, 6.7584622], [9.6455333, 9.7584612]])
    # y = np.array([[0], [1], [0], [0], [1], [1]])
    x = np.array(x)
    y = np.array(y)
    model = None
    for step in range(eval(epoch)):
        a, da, x = myNeuralNet.forward_propagation(x)
        myNeuralNet.loss(a, y)
        w, b = myNeuralNet.backward_propagation(a, da, x, y)
        model = myNeuralNet.update_parameters(w, b)
        print("finished {} epoch".format(step+1))
    print(myNeuralNet.predict(x))
    print(model)
