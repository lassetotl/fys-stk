### Classes for Feed Forward Neural Networks with implemented backpropagation
### for linear regression and classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from p2_functions import MSE, FrankeFunction, create_X, sigmoid, RELU

from sklearn.metrics import accuracy_score

np.random.seed(0)

class FFNN_Regression:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=20,
            n_categories=1,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            activation = 'Sigmoid'):

        self.X_data = X_data
        self.Y_data = Y_data
        self.activation = activation

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        #storing loss every epoch
        self.MSE_ = np.zeros(self.epochs)

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)*0.01
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)*0.01
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self): # modified for regression
        # weighted sum of inputs to the hidden layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights)

        # activation in the hidden layer
        if self.activation == 'Sigmoid':
            self.a_h = sigmoid(self.z_h)
        elif self.activation == 'Leaky RELU':
            self.a_h = RELU(self.z_h)
        elif self.activation == 'RELU':
            self.a_h = RELU(self.z_h, a = 0)
        elif self.activation == 'Tanh':
            self.a_h = np.tanh(self.z_h)

        # weighted sum of inputs to the output layer
        self.z_o = np.matmul(self.a_h, self.output_weights)

    def backpropagation(self):
        error_output = (self.z_o - self.Y_data)/self.Y_data.shape[0]
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def train(self):
        for i in range(self.epochs):
            self.feed_forward()
            self.backpropagation()
            self.MSE_[i] = MSE(self.Y_data, self.z_o)
        # Plotting MSE for each epoch step
        epoch = np.linspace(0, self.epochs, self.epochs)
        plt.plot(epoch, self.MSE_, lw = 3,
        label = f'{self.activation}, $\eta = {self.eta}$, $\lambda = {self.lmbd}$, '+'$MSE_{min}$ = '+f'{round(np.min(self.MSE_),3)}')
        return epoch, self.MSE_

class FFNN_Classification:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=20,
            n_categories=2,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            activation = 'Sigmoid'):

        self.X_data = X_data
        self.Y_data = Y_data
        self.activation = activation

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        #storing loss and accuracy every epoch
        self.cost_ = np.zeros(self.epochs)
        self.acc_ = np.zeros(self.epochs)

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)*0.01
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)*0.01
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self): # modified for regression
        # weighted sum of inputs to the hidden layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights)

        # activation in the hidden layer
        if self.activation == 'Sigmoid':
            self.a_h = sigmoid(self.z_h)
        elif self.activation == 'Leaky RELU':
            self.a_h = RELU(self.z_h)
        elif self.activation == 'RELU':
            self.a_h = RELU(self.z_h, a = 0)
        elif self.activation == 'Tanh':
            self.a_h = np.tanh(self.z_h)

        # weighted sum of inputs to the output layer
        self.z_o = np.matmul(self.a_h, self.output_weights)
        # softmax output, gives shape(prob) = shape(Y)
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = np.exp(self.z_o)
        self.z_o = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def backpropagation(self):
        error_output = (self.z_o - self.Y_data)/(self.z_o*(1 - self.z_o))
        error_output = error_output/self.Y_data.shape[0]
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def train(self):
        for i in range(self.epochs):
            self.feed_forward()
            self.backpropagation()
                # save accuracy for each step
            predict = np.argmax(self.z_o, axis=1)
            self.acc_[i] = accuracy_score(predict, self.Y_data)
            self.cost_[i] = -np.sum(self.Y_data*np.log(self.z_o) + (1-self.Y_data)*np.log(1-self.z_o))/self.Y_data.shape[0]
        # Plotting accuracy for each epoch step
        epoch = np.linspace(0, self.epochs, self.epochs)
        plt.plot(epoch, self.acc_, lw = 3,
        label = f'{self.activation}, $\eta = {self.eta}$, $\lambda = {self.lmbd}$, '+'$MSE_{min}$ = '+f'{round(np.max(self.acc_),3)}')
        return epoch, self.acc_, self.z_o, self.cost_
