import numpy as np
def grad(X, y, ytilde):
    m = X.shape[0]

    gradW = - 1 / m * X.T @ (y.reshape(ytilde.shape) - ytilde)
    gradb = 1 / m * np.sum((y.reshape(ytilde.shape) - ytilde))

    return gradW, gradb
