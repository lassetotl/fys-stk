import numpy as np
from gradient import *

def plain_gradient_adagrad(learn_rate, n_iterations, X, y, Beta, lmbd):
    n = np.shape(X)[0]
    epsilon  = 1e-13
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    for i in range(n_iterations):
        for j in range(i):
            g = gradient(n, X, y, Beta, lmbd)
            G = g @ g.T
            new_learn_rate = learn_rate / (epsilon + np.sqrt(np.diagonal(G)))
            G_inverse = np.c_[new_learn_rate]
            #learn_rate = new_learn_rate
            update = np.multiply(G_inverse, g)
            Beta = Beta - update
    return Beta
