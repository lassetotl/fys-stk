import numpy as np
from gradient import *

def plain_gradient_adagrad(learn_rate, n_iterations, X, y, Beta):
    n = np.shape(X)[0]
    epsilon  = 1e-13
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    for i in range(n_iterations):
        g = gradient(n, X, y, Beta)
        G = G + (g @ g.T)
        G_inverse = np.c_[learn_rate / ( epsilon + np.sqrt(np.diagonal(G)) )]
        update = np.multiply(G_inverse, g)
        Beta = Beta - update
    return Beta
