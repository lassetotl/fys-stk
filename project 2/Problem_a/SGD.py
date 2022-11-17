import numpy as np
from gradient import *
from learning_schedule import *

def SGD(M, n_epoch, learn_rate, X, y, Beta, lmbd):
    n = np.shape(X)[0]
    m = int(n/M) #number of minibatches
    t0 = 1.0
    t1 = 10
    gamma_j = t0/t1
    j = 0
    for epoch in range(1, n_epoch + 1):
        for i in range(m):
            random_index = M * np.random.randint(m) #Pick the random index
            X_batch = X[random_index:random_index+M]
            y_batch = y[random_index:random_index+M]
            g = (1/M) * gradient(n, X_batch, y_batch, Beta, lmbd)
            learn_rate = learning_schedule(epoch*m + i, t0, t1)
            Beta = Beta - learn_rate*g
    return Beta
