from sigmoid import *
from gradient import *
def SGD_train(M, n_epochs, X, y, w, b, lmbd_, learning_rate_):
    n = np.shape(X)[0] #number of datapoints
    m = int(n/M) #number of minibatches

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M * np.random.randint(m)
            X_batch = X[random_index:random_index+M]
            y_batch = y[random_index:random_index+M]

            var = X_batch @ w + b
            P = sigmoid(var) #probability
            grad_w, grad_b = gradient(X_batch, y_batch, P)

            grad_w += lmbd_ * w
            grad_b += lmbd_ * b

            w -= learning_rate_ * grad_w
            b -= learning_rate_ * grad_b

            return w, b
