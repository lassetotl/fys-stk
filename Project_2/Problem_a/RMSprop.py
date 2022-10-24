import numpy as np
import matplotlib.pyplot as plt


#Generating data:
np.random.seed(10)
n = 100
x = 2 * np.random.randn(n,1)
y = -0.5*x**3 + 1.5*x**2 + 1.5*x + 3*np.random.randn(n,1)

X = np.c_[np.ones((n, 1)), x, x**2, x**3]
H = (2.0/n) * X.T @ X
EigVal, EigVec = np.linalg.eig(H)

def gradient(X, y, Beta, n):
    return (2.0/n) * X.T @ (X @ Beta - y)


n_epochs = 50
M = 5
Beta = np.random.randn(4,1)
learn_rate = 0.01
m = int(n/M) #number of minibatches
epsilon  = 1e-8

rho = 0.99


for epoch in range(n_epochs):
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    for i in range(m):
        random_index = M*np.random.randint(m)
        X_minibatch = X[random_index:random_index + M]
        y_minibatch = y[random_index:random_index + M]
        g = (1/M) * gradient(X_minibatch, y_minibatch, Beta, n)

        prev_G = G
        G = G + (g @ g.T)

        G_new = (rho * prev_G + (1 - rho) * G)

        G_inverse = np.c_[learn_rate / (epsilon + np.sqrt(np.diagonal(G_new)))]
        update = np.multiply(G_inverse , g)
        Beta = Beta - update

print(Beta)
