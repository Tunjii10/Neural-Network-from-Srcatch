# References
# Random Seed Initialization (line 17), Available: https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch
# Self.layer_input_output_pairs loop (line 38-40), Available https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch. Changes: Implemented in class structure
# Self.network initialization loop (line 45-48), Available https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch. Changes: Implemented in class structure, changed initialization of bias to zeros values
# Return statement in Predict function (line 140), Available https://github.com/pbrebner/binary-classification-from-scratch/blob/main/binary_breast_cancer_diagnosis.ipynb. Changes: Used a python dictionary for storing y_pred
# Accuracy Function (line 144-146), Available https://www.kaggle.com/code/jantinbergen/challenge-1-bletchley-three-layer-nnetwork. Changes: summed over a specified axis
# Shuffle dataset in Mini Bacth Gradient Descent(line 205-206), AVailabe https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison [andrea m post]
import os
import time
import numpy as np
import pandas as pd
from Activation_Loss_Functions import activations


class Ann:
    def __init__(self, neurons_in_layers, activations_loss, type_learning_rate, learning_rate, loss_function):
        # *Available :https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch (line 17)
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
        self.type_learning_rate = eval(type_learning_rate)

        # initialize  activation class for loss function used
        self.init_loss_activate = activations.ActivationsLoss(
            self.loss_function)

        # loop through neuron layers to get structure pair for weight dimensions
        # AVailable : https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch (line 38-40)
        for i in range(len(self.neurons_in_layers)-1):
            self.layer_input_output_pairs.append(
                (self.neurons_in_layers[i], self.neurons_in_layers[i+1]))

        # create  network by initializing weights and bias using structure pairs
        # AVailable : https://github.com/bp249507/Multi-Layer-Neural-Network-from-scratch (line 45-48)
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
            0, a, y)  # call loss function in activation class
        print("Loss  = {}".format(loss))
        return loss

    def backward_propagation(self, a, da, datax, y):
        dz = {}
        db = {}
        dw = {}
        # loop backwards through length 'a' == length of forward activations gotten in forward pass
        for x in range(len(a)-1, -1, -1):
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

    def update_parameters(self, dw, db, iter):
        # for each layer update weights and biases
        for layer in range(len(self.network)):
            # for constant learning rate
            if self.type_learning_rate[0] == 'Const':
                self.network[layer]['W'] = self.network[layer]['W'] - \
                    self.learning_rate*dw[layer]
                self.network[layer]['b'] = self.network[layer]['b'] - \
                    self.learning_rate*db[layer]
            # for learning rate scheduler. We implemented a time base scheduler based on the epoch
            elif self.type_learning_rate[0] == 'TBD':
                self.learning_rate = (1 / (1 + (self.learning_rate * iter)))
                self.network[layer]['W'] = self.network[layer]['W'] - \
                    self.learning_rate*dw[layer]
                self.network[layer]['b'] = self.network[layer]['b'] - \
                    self.learning_rate*db[layer]
            else:
                raise Exception(
                    "Invalid learning rate")
        return self.network

    def predict(self, x):
        # predict funtion
        a_, _, _ = self.forward_propagation(x)
        # * AVAilable: https://github.com/pbrebner/binary-classification-from-scratch/blob/main/binary_breast_cancer_diagnosis.ipynb (line140)
        return a_[len(a_)-1].round()

    def accuracy(self, y_pred, y):  # Accuracy function
        # *AVailable : https://www.kaggle.com/code/jantinbergen/challenge-1-bletchley-three-layer-nnetwork (line 144-146)
        error = np.abs(y_pred-y)
        correct = self.m-np.sum(error, axis=None)
        correct_percentage = correct/self.m * 100
        return correct_percentage

    def store_data(self, path, accuracy, training_time, loss, epochs, learning_rate, number_hidden_nodes):
        # Store model hyperparameter values for later use
        array = np.array([accuracy, training_time, loss, epochs,
                         learning_rate, number_hidden_nodes])
        np.save(path+"/Results/{}%accuracy-{}-epoch-{}seconds-{}-layers.np".format(
            accuracy, epochs, training_time, number_hidden_nodes), array)
        return


def pipeline(neurons_in_layers, activations, type_learning_rate, learning_rate, epoch, loss_function, grad_descent, batch_size):
    '''
        Step by step working of the neural network
        for each epooch = forward propagate -> back propagate -> update weight and biases
    '''
    start_time = time.time()  # start time
    myNeuralNet = Ann(neurons_in_layers, activations, type_learning_rate,
                      learning_rate, loss_function)  # call neuron class
    path = os.getcwd()
    data = pd.read_csv(path+"/data/wdbc.data", sep=",", header=None)
    # seperate data into x and y values
    x = data.iloc[:, 2:]
    y = data.iloc[:, 1]
    # replace categorical values to binaries O and 1
    y = y.replace({'M': 1, 'B': 0})
    # turn x and y values to an numpy array
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(len(y), 1)  # change y to 2 dimensional array
    # Initialize parameters
    model = None
    accuracy = 0
    loss = 0
    grad_descent = eval(grad_descent)
    if grad_descent[0] == 'BGD':  # if batch gradient descent
        for step in range(eval(epoch)):  # epoch loop
            a, da, x = myNeuralNet.forward_propagation(x)
            loss = myNeuralNet.loss(a, y)
            w, b = myNeuralNet.backward_propagation(a, da, x, y)
            model = myNeuralNet.update_parameters(w, b, step)
            print("finished {} epoch".format(step+1))
        y_pred = myNeuralNet.predict(x)  # predict y with x values
        accuracy = myNeuralNet.accuracy(y_pred, y)  # calculate accuracy
        print("accuracy = {}".format(accuracy))
    elif grad_descent[0] == 'MBGD':  # if mini batch gradient descent

        batch_size = eval(batch_size)

        # check if batch size
        if len(x) < batch_size:
            raise Exception(
                "Batch_size exceeds no of samples ")  # check if batch size greater than  input samples
        batch_pointer = 0  # pointer to reloop batch when we are reaching the end of batches
        for step in range(eval(epoch)):
            # epoch loop
            if len(x) == len(y):
                # * Available : https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison (line 205-206)
                c = np.arange(len(x))
                np.random.shuffle(c)
                x = np.array(x[c])
                y = np.array(y[c])
            if batch_pointer*batch_size+batch_size > len(x):
                batch_pointer = 0  # initialize to zero if pointer above length of data
            start = batch_pointer*batch_size
            # slice of batch from x data
            x_batch = x[start:start + batch_size, :]
            # slice of batch from y data
            y_batch = y[start:start + batch_size, :]
            a, da, x_batch = myNeuralNet.forward_propagation(x_batch)
            loss = myNeuralNet.loss(a, y_batch)
            w, b = myNeuralNet.backward_propagation(a, da, x_batch, y_batch)
            model = myNeuralNet.update_parameters(w, b, step)
            batch_pointer += 1
            print("finished {} epoch".format(step+1))
        y_pred = myNeuralNet.predict(x)  # predict y
        accuracy = myNeuralNet.accuracy(y_pred, y)  # calculate accuracy
        print("accuracy = {}".format(accuracy))
    elif grad_descent[0] == 'SGD':  # if stochastic gradient descent
        for step in range(eval(epoch)):
            if len(x) == len(y):
                random_index = np.random.randint(
                    len(x))  # select random number from x
            x_point_data = x[[random_index], :]  # x data point
            y_point_data = y[[random_index], :]  # y data point
            a, da, x_point_data = myNeuralNet.forward_propagation(x_point_data)
            loss = myNeuralNet.loss(a, y_point_data)
            w, b = myNeuralNet.backward_propagation(
                a, da, x_point_data, y_point_data)
            model = myNeuralNet.update_parameters(w, b, step)
            print("finished {} epoch".format(step+1))
        y_pred = myNeuralNet.predict(x)  # predict
        accuracy = myNeuralNet.accuracy(y_pred, y)  # accuracy
        print("accuracy = {}".format(accuracy))
    else:
        print("Gradient Descent Algorithm invalid")
    time_stop = time.time()  # stop timer

    print("Time elapsed is {}s".format(float(time_stop-start_time)))
    neurons_in_layers = eval(neurons_in_layers)
    myNeuralNet.store_data(path, accuracy, time_stop-start_time, loss, epoch,
                           learning_rate, len(neurons_in_layers)-2)  # store data
