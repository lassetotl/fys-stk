import numpy as np
import matplotlib.pyplot as plt

# #Own libraries
# from data import DataPoints
# from gradient import Gradient_function
#
#
# class Adam:
#     def __init__(self, n, n_epochs, learn_rate, M, momentum):
#         self.n = n
#         self.n_epochs = n_epochs
#         self.learn_rate = learn_rate
#         self.M = M
#         self.momentum = momentum
#         self.X = DataPoints(self.n).output()
#
#         self.Beta = np.random.randn(np.shape(self.X)[1], 1)
#         print(self.Beta)

#For Adam optimization, we simply add RMSpop method and Gradient descent with momentum method together
#Adam(100, 50, 0.001, 5, 0.3)
# Beta = np.random.randn(4,1)
# print(Beta)
m = int(n/M) #number of minibatches
epsilon  = 1e-8

rho = 0.99

change = 0.0



for epoch in range(n_epochs):
    G = np.zeros(shape=(np.shape(X)[1],np.shape(X)[1]))
    for i in range(m):
        random_index = M*np.random.randint(m)
        X_minibatch = X[random_index:random_index + M]
        y_minibatch = y[random_index:random_index + M]
        g = (1/M) * Gradient_function(n, X_minibatch, y_minibatch, Beta).regular_gradient()

        prev_G = G
        G = G + (g @ g.T)

        G_new = (rho * prev_G + (1 - rho) * G)

        G_inverse = np.c_[learn_rate / (epsilon + np.sqrt(np.diagonal(G_new)))]
        new_change = np.multiply(G_inverse , g) + momentum*change
        change = new_change
        Beta = Beta - new_change

y_predict = X @ Beta

plt.plot(x, y, "o", label="real data")
plt.plot(x, y_predict, "o", label="Prediction")
plt.legend()
plt.show()
