import numpy as np
from learning_schedule import *
from gradient import *

def SGD_momentum(M, n_epoch, learn_rate, X, y, Beta):
    n = np.shape(X)[0]
    m = int(n/M) #number of minibatches
    t0 = 1.0
    t1 = 10
    gamma_j = t0/t1
    j = 0
    change = 0.0
    momentum = 0.3
    for epoch in range(1, n_epoch + 1):
        for i in range(m):
            random_index = M * np.random.randint(m) #Pick the random index
            X_batch = X[random_index:random_index+M]
            y_batch = y[random_index:random_index+M]
            g = (1/M) * gradient(n, X_batch, y_batch, Beta)
            learn_rate = learning_schedule(epoch*m + i, t0, t1)
            new_change = learn_rate*g + momentum*change
            Beta = Beta - new_change
            change = new_change
    return Beta
