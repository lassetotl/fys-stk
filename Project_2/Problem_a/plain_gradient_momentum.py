import numpy as np
from gradient import *

def plain_gradient_momentum(learn_rate, N_iterations, X, y, Beta):
    n = np.shape(X)[0]
    change = 0.0
    momentum = 0.3
    for i in range(N_iterations):
        g = gradient(n, X, y, Beta)
        new_change = learn_rate*g + momentum*change
        Beta = Beta - new_change
        change = new_change
    return Beta
