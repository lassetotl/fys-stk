import numpy as np
import matplotlib.pyplot as plt
from gradient import *
from learning_schedule import *



def SGD_ADAM(M, n_epochs, learn_rate, X, y, Beta):
    n = np.shape(X)[0]
    m = int(n/M) #number of minibatches
    epsilon  = 1e-8
    rho = 0.99
    momentum = 0.3
    change = 0.0
    t0 = 1.0
    t1 = 10


    for epoch in range(n_epochs):
        G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
        for i in range(m):
            random_index = M*np.random.randint(m)
            X_minibatch = X[random_index:random_index + M]
            y_minibatch = y[random_index:random_index + M]
            g = (1/M) * gradient(n, X_minibatch, y_minibatch, Beta)
            prev_G = G
            G = G + (g @ g.T)
            G_new = (rho * prev_G + (1 - rho) * G)
            learn_rate = learning_schedule(epoch*m + i, t0, t1)
            G_inverse = np.c_[learn_rate / (epsilon + np.sqrt(np.diagonal(G_new)))]
            new_change = np.multiply(G_inverse , g) + momentum*change
            change = new_change
            Beta = Beta - new_change
    return Beta
