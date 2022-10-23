import numpy as np
import matplotlib.pyplot as plt
from autograd import grad


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

#learn_rate = 1.0/np.max(EigVal)
n_epochs = 50
M = 5
m = int(n/M) #number of minibatches
Beta = np.random.randn(4,1)
learn_rate = 0.01
epsilon  = 1e-8

change = 0.0
momentum = 0.3

for epoch in range(n_epochs):
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    G = np.zeros(shape=(4,4))
    for i in range(m):
        random_index = M*np.random.randint(m)
        X_minibatch = X[random_index:random_index + M]
        y_minibatch = y[random_index:random_index + M]
        g = (1/M) * gradient(X_minibatch, y_minibatch, Beta, n)
        G = G + (g @ g.T)
        G_inverse = np.c_[learn_rate / ( epsilon + np.sqrt(np.diagonal(G)) )]
        new_change = np.multiply(G_inverse, g) + momentum*change
        change = new_change
        Beta = Beta - new_change
print(Beta)
# test_x = np.linspace(-4, 4, n)
# test_X = np.c_[np.ones((n, 1)), test_x, test_x**2, test_x**3]
y_predict = X @ Beta

plt.plot(x, y, "o", label="real data")
plt.plot(x, y_predict, "o", label="Prediction")
plt.legend()
plt.show()
