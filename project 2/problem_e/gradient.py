import numpy as np

def gradient(X, y, ytilde):
    n = X.shape[0]
    grad_w = -1/n * X.T @ (y.reshape(ytilde.shape) - ytilde)
    grad_b = 1/n * np.sum((y.reshape(ytilde.shape) - ytilde))

    return grad_w, grad_b
