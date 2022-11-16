import numpy as np
import matplotlib.pyplot as plt
from gradient import *

def SGD_adagrad_momentum(M, n_epochs, learn_rate, X, y, Beta):
    n = np.shape(X)[0]
    m = int(n/M) #number of minibatches
    epsilon  = 1e-8
    change = 0.0
    momentum = 0.3

    for epoch in range(n_epochs):
        G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
        for i in range(m):
            random_index = M*np.random.randint(m)
            X_minibatch = X[random_index:random_index + M]
            y_minibatch = y[random_index:random_index + M]
            g = (1/M) * gradient(n, X_minibatch, y_minibatch, Beta)
            G = G + (g @ g.T)
            G_inverse = np.c_[learn_rate / ( epsilon + np.sqrt(np.diagonal(G)) )]
            new_change = np.multiply(G_inverse, g) + momentum*change
            change = new_change
            Beta = Beta - new_change
    return Beta
