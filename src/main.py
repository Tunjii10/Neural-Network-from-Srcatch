import numpy as np
from NeuralNet import ann


def pipeline():
    neurons_in_layers = str(input(
        "What number of neuron(s) do you want in each layer e.g 2 neurons input layer, 1 neuron 1st hidden layer, 3 neurons second hidden layer, 2 output neurons = [2,1,3,2] \n Array must have minimum length of three:"))
    hidden_activations = str(input(
        "What number of type of activation function dou you want in the hidden and output layer e.g Relu in first hidden layer, Sigmoid in second layer, Cross Entropy in last layer = ['Relu', 'Sig', 'CE'] \n *********************** \n Available function codes ['Sig', 'Relu', 'Tanh', 'CE'] \n ********************* \n:"))

    myNeuralNet = ann.Ann(neurons_in_layers, hidden_activations, 1, 1, 1, 1, 1)
    data = np.array([[1, 2]])
    myNeuralNet.forward_propagation(data)


if __name__ == "__main__":
    pipeline()
