import numpy as np
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
                if back_a != "no_back_propagate":  # for last layer activation no derivative taken
                    da[y] = back_a
        return a, da, x  # outputs forward activation, derivative of activation and x input

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


def pipeline(neurons_in_layers, activations, learning_rate, epoch, c, d):
    '''
        Step by step working of the neural network
        for each epooch = forward propagate -> back propagate -> update weight and biases
    '''
    myNeuralNet = Ann(neurons_in_layers, activations,
                      learning_rate, c, d)

    x = np.array([[1, 2], [3, 5]])
    y = np.array([[1], [2]])
    model = None
    for step in range(eval(epoch)):
        a, da, x = myNeuralNet.forward_propagation(x)
        w, b = myNeuralNet.backward_propagation(a, da, x, y)
        model = myNeuralNet.update_parameters(w, b)
        print("finished {} epoch".format(step+1))
    print(model)
