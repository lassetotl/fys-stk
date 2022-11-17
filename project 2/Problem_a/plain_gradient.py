import numpy as np
from gradient import *

def plain_gradient(learn_rate, N_iterations, X, y, Beta, lmbd):
    n = np.shape(X)[0]
    for i in range(N_iterations):
        g = gradient(n, X, y, Beta, lmbd)
        Beta = Beta - learn_rate * g
    return Beta
