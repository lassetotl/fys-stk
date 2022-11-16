import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from SGD_train import *

np.random.seed(0)

#Defining the data we are using, and splitting it
data, target = load_breast_cancer(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

def analysis(X, y):
    #Defining the variables
    n, n_features = X.shape
    b = 0
    eta = 0.3
    w = np.zeros([n_features, 1])
    M = 10 #batch size
    n_epochs = 50
    n_iterations = n // M
    #lmbda = 0.1

    learning_rate = np.logspace(-3, 0, 10)
    lmbda = np.logspace(-3, 0, 10)

    train_check = np.zeros([len(learning_rate), len(lmbda)])
    test_check = np.zeros([len(learning_rate), len(lmbda)])

    ytilde = train(n, n_epochs, n_iterations, M, X, y, w, b, lmbda, eta)
analysis(train_data, train_target)
