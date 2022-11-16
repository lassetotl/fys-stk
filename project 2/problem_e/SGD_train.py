import numpy as np
from grad import *
from sigmoid import *
def train(n, n_epochs, n_iterations, M, X, y, w, b, lmbda, eta):
    """Trains the logistic regressor weights
            using Stochastic gradient descent (Ridge or OLS depending on lmbda value)

    """
    data_indices = np.arange(n)
    #errors = []
    for epoch in range(n_epochs):
        for j in range(n_iterations):
            # pick datapoints with replacement
            chosen_datapoints = np.random.choice(
                data_indices, size=M, replace=False
            )

            # minibatch training data
            X_k = X[chosen_datapoints]
            y_k = y[chosen_datapoints]

            probability = sigmoid(X_k @ w + b)
            dw, db = grad(X_k, y_k, probability)


            # dw += lmbda * w
            # db += lmbda * b

            w -= eta * dw
            b -= eta * db

        ytilde = sigmoid(X @ w + b)
        #errors.append(self.cost_func(self.sigmoid(X @ w + b)))
    return ytilde#, errors
