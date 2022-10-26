import numpy as np
from NeuralNet import ann


def start():
    # neurons_in_layers = str(input(
    #     "\n ------------------------ \n What number of neuron(s) do you want in each layer e.g 2 neurons input layer, 1 neuron 1st hidden layer, 3 neurons second hidden layer, 1 output neurons = [2,1,3,1] \n Array must have minimum length of three \n Output layerfor the model implementation must have one neuron(binary classification) and input layer must be 30 for breast cancer dataset \n:"))
    # activations = str(input(
    #     "\n ------------------------ \n What activation functions  do you want in the hidden and output layer e.g Relu in first hidden layer, Sigmoid in second layer, Sigmoid in last layer/Output layer = ['Relu', 'Sig', 'CE'] \n *********************** \n Available function codes ['Sig'=Sigmoid, 'Relu'=Relu, 'Tanh'=Tanh] \n *****************\n This implementation uses Sigmoid as its activation for the output. Therefore Sigmoid must be used in the last layer for this model to work appropriately. \n ********************* \n:"))
    # type_learning_rate = str(input(
    #     "\n ------------------------ \n What type of learning rate e.g constant or time based decay =>> ['Const'] or ['TBD'] \n ********************* \n:"))
    # learning_rate = (
    #     input("\n ------------------------ \n What is the learning rate(for 'Const')/ decay(for 'TBD') eg. 0.01:\n"))
    # epoch = input(
    #     "\n ------------------------ \n What number of iterations do you want the model to learn for i.e epoch eg. 1,300,400:\n")
    # loss_function = str(input(
    #     "\n ------------------------ \n What is your loss function eg. Binary Cross Entropy = ['BCE']:\n *********** \n Binary Cross Entropy only implemented: \n"))
    # gradient_descent = str(input(
    #     "\n ------------------------ \n What type of gradient descent eg. batch gradient desc = ['BGD'] \n Here are list of grad. desc. available Batch = 'BGD', MiniBatch = 'MBGD', Stochastic = 'SGD' \n Note: Gradient DEscent must be in array and only one selected:\n"))
    # batch_size = input(
    #     "\n ------------------------ \n What is your batch size eg. for BGD(full dataset) and SGD(random data point) you can input 0, for MGD input an integer numer eg. 5,32,4:\n")
    # ann.pipeline(neurons_in_layers, activations, type_learning_rate,
    #              learning_rate, epoch, loss_function, gradient_descent, batch_size)
    ann.pipeline("[30, 50, 40, 30, 1]", "['Tanh', 'Tanh', 'Relu', 'Tanh']", "['TBD']",
                 "0.0005", "20", "['BCE']", "['BGD']", "40")


if __name__ == "__main__":
    start()
