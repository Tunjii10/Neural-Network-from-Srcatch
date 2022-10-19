import numpy as np
from ActivationFunctions import activations


class Ann:
    def __init__(self, neurons_in_layers, hidden_activations, output_activations, learning_rate, epoch, loss_function, grad_descent):
        np.random.seed(0)
        self.network = dict()
        self.neurons_in_layers = eval(neurons_in_layers)
        self.activation = eval(hidden_activations)
        self.layer_coeff = []
        for i in range(len(self.neurons_in_layers)-1):
            self.layer_coeff.append(
                (self.neurons_in_layers[i], self.neurons_in_layers[i+1]))
        for c in range(len(self.layer_coeff)):
            Weight = np.random.randn(
                self.layer_coeff[c][0], self.layer_coeff[c][1])
            bias = np.zeros((1, self.layer_coeff[c][1]))
            self.network[c+1] = {'W': Weight, 'b': bias}
        self.network_length = len(self.network)
        print("network bias and weights \n >>>>>>>> \n {}".format(self.network))

    def forward_propagation(self, data):
        if self.neurons_in_layers[0] != data.shape[1]:
            raise Exception(
                "Your specified inputs no doesnt match data provided")
        elif len(self.activation) != self.network_length:
            raise Exception(
                "Activation functions specified dont match layers provided")
        else:
            a = {0: data}
            da = {}
            for i in range(self.network_length-1):
                z = a[i].dot(self.network[i+1]['W']) + self.network[i+1]['b']
                a[i+1], da[i +
                           1] = activations.Activations(self.activation[i], z)
            # z = a[l_-1].dot(model[l_-1]['W']) + model[l_-1]['b']
            # a[l_] = softmax(z)
            # return a, da
