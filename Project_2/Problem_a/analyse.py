import numpy as np
import matplotlib.pyplot as plt
from data import *
from plain_gradient import *
from plain_gradient_momentum import *
from SGD import *
from SGD_momentum import *
from plain_gradient_adagrad import *
from plain_gradient_adagrad_momentum import *
from SGD_adagrad import *
from SGD_adagrad_momentum import *

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

n = 100 #number of datapoints
x, y, X, max_EigVal = DataPoints(n).output()


#Variables:
learn_rate = 1/max_EigVal
n_iterations = 70
Beta_start = np.random.randn(np.shape(X)[1],1)
M = 5 #Size of each minibatch
n_epoch = 40 #number of epochs


"""
Analysis of plain gradient descents:
"""
N = np.zeros(n_iterations)
MSE1 = np.zeros(n_iterations)
MSE2 = np.zeros(n_iterations)
MSE3 = np.zeros(n_iterations)
MSE4 = np.zeros(n_iterations)
for i in range(n_iterations):
    Beta1 = plain_gradient(learn_rate, i, X, y, Beta_start )
    y_pred1 = X @ Beta1
    MSE1[i] = MSE(y,y_pred1)
    Beta2 = plain_gradient_momentum(learn_rate, i, X, y, Beta_start)
    y_pred2 = X @ Beta2
    MSE2[i] = MSE(y,y_pred2)
    Beta3 = plain_gradient_adagrad(learn_rate, i, X, y, Beta_start)
    y_pred3 = X @ Beta3
    MSE3[i] = MSE(y,y_pred3)
    Beta4 = plain_gradient_adagrad_momentum(learn_rate, i, X, y, Beta_start)
    y_pred4 = X @ Beta4
    MSE4[i] = MSE(y,y_pred4)




    N[i] = i

plt.plot(N, MSE1, label="plain gradient")
plt.plot(N, MSE2, label="plain gradient with momentum")
plt.plot(N, MSE3, label="plain gradient with adagrad")
plt.plot(N, MSE4, "--r", label="plain gradient with adagrad and momentum")
plt.legend()
plt.show()
"""
Analysis of stochastic gradient descents
"""
learn_rate = 0.01
N = np.zeros(n_epoch)
MSE1 = np.zeros(n_epoch)
MSE2 = np.zeros(n_epoch)
MSE3 = np.zeros(n_epoch)
MSE4 = np.zeros(n_epoch)

for i in range(n_epoch):
    Beta1 = SGD(M, i, learn_rate, X, y, Beta_start)
    y_pred1 = X @ Beta1
    MSE1[i] = MSE(y,y_pred1)

    Beta2 = SGD_momentum(M, i, learn_rate, X, y, Beta_start)
    y_pred2 = X @ Beta2
    MSE2[i] = MSE(y,y_pred2)

    Beta3 = SGD_adagrad(M, i, learn_rate, X, y, Beta_start)
    y_pred3 = X @ Beta3
    MSE3[i] = MSE(y,y_pred3)

    Beta4 = SGD_adagrad_momentum(M, i, learn_rate, X, y, Beta_start)
    y_pred4 = X @ Beta4
    MSE4[i] = MSE(y,y_pred4)


    N[i] = i

plt.plot(N, MSE1, label="simple SDG")
plt.plot(N, MSE2, label="simple SDG with momentum")
plt.plot(N, MSE3, label="simple SDG with adagrad")
plt.plot(N, MSE4, label="simple SDG with adagrad and momentum")
plt.legend()
plt.show()
