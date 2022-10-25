import numpy as np
from NeuralNet import ann


def start():
    # neurons_in_layers = str(input(
    #     "What number of neuron(s) do you want in each layer e.g 2 neurons input layer, 1 neuron 1st hidden layer, 3 neurons second hidden layer, 1 output neurons = [2,1,3,1] \n Array must have minimum length of three \n Output layer must have one neuron i.e binary classification:"))
    # activations = str(input(
    #     "What number of activations and function types(loss and activations) do you want in the hidden and output layer e.g Relu in first hidden layer, Sigmoid in second layer, Cross Entropy in last layer/Output layer = ['Relu', 'Sig', 'CE'] \n *********************** \n Available function codes ['Sig'=Sigmoid, 'Relu'=Relu, 'Tanh'=Tanh, 'CE=Cross Entropy', 'SM'=softmax] \n ********************* \n:"))
    # learning_rate = input("what is the learning rate eg. 1,0.1,0.5:\n")
    # epoch = input("what number of epoch do you need eg. 1,300,400:\n")
    # loss_function = input(
    #     "what is your loss function eg. Binary Cross Entropy = ['BCE']:\n")
    # ann.pipeline(neurons_in_layers, activations,
    #              learning_rate, epoch, loss_function, 1)
    ann.pipeline("[30, 1, 1]", "['Sig', 'Sig']",
                 "0.0005", "500", "['BCE']", 1)


if __name__ == "__main__":
    start()
