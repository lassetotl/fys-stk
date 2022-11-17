import numpy as np
import matplotlib.pyplot as plt
from gradient import *


def SGD_adagrad(M, n_epochs, learn_rate, X, y, Beta, lmbd):
    n = np.shape(X)[0]
    epsilon  = 1e-8
    m = int(n/M) #number of minibatches
    for epoch in range(n_epochs):
        G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
        #G = np.zeros(shape=(4,4))
        for i in range(m):
            random_index = M*np.random.randint(m)
            X_minibatch = X[random_index:random_index + M]
            y_minibatch = y[random_index:random_index + M]
            g = (1/M) * gradient(n, X_minibatch, y_minibatch, Beta, lmbd)
            G = G + (g @ g.T)
            G_inverse = np.c_[learn_rate / ( epsilon + np.sqrt(np.diagonal(G)) )]
            update = np.multiply(G_inverse, g)
            Beta = Beta - update
    return Beta
