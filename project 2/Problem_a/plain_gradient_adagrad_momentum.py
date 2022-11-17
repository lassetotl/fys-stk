import numpy as np
from gradient import *

def plain_gradient_adagrad_momentum(learn_rate, n_iterations, X, y, Beta, lmbd):
    n = np.shape(X)[0]
    epsilon  = 1e-13
    change = 0.0
    momentum = 0.3
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    for i in range(n_iterations):
        for j in range(i):
            g = gradient(n, X, y, Beta, lmbd)
            G = g @ g.T
            new_learn_rate = learn_rate / (epsilon + np.sqrt(np.diagonal(G)))
            G_inverse = np.c_[new_learn_rate]
            new_change = np.multiply(G_inverse, g) + momentum*change
            change = new_change
            #learn_rate = new_learn_rate
            Beta = Beta - new_change
    return Beta
