import numpy as np
from NeuralNet import ann


def start():
    neurons_in_layers = str(input(
        "What number of neuron(s) do you want in each layer e.g 2 neurons input layer, 1 neuron 1st hidden layer, 3 neurons second hidden layer, 2 output neurons = [2,1,3,2] \n Array must have minimum length of three:"))
    activations = str(input(
        "What number of activations and function types(loss and activations) do you want in the hidden and output layer e.g Relu in first hidden layer, Sigmoid in second layer, Cross Entropy in last layer/Output layer = ['Relu', 'Sig', 'CE'] \n *********************** \n Available function codes ['Sig'=Sigmoid, 'Relu'=Relu, 'Tanh'=Tanh, 'CE=Cross Entropy', 'SM'=softmax] \n ********************* \n:"))
    learning_rate = input("what is the learning rate eg. 1,0.1,0.5:\n")
    epoch = input("what number of epoch do you need eg. 1,300,400:\n")
    ann.pipeline(neurons_in_layers, activations, learning_rate, epoch, 1, 1)
    # ann.pipeline("[2, 3, 1]", "['Sig', 'SM']",
    #              "0.1", "500", 1, 1)


if __name__ == "__main__":
    start()
