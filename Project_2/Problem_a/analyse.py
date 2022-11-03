import numpy as np
import matplotlib.pyplot as plt
from data import *
from plain_gradient import *
from plain_gradient_momentum import *
from SGD import *
from SGD_momentum import *
from plain_gradient_adagrad import *

n = 100 #number of datapoints
x, y, X, max_EigVal = DataPoints(n).output()


#Variables:
learn_rate = 1/max_EigVal
n_iterations = 40
Beta_start = np.random.randn(np.shape(X)[1],1)
M = 5 #Size of each minibatch
n_epoch = 10 #number of epochs

#Plain gradient:
Beta1 = plain_gradient(learn_rate, n_iterations, X, y, Beta_start )

#Plain gradient with momentum
Beta2 = plain_gradient_momentum(learn_rate, n_iterations, X, y, Beta_start)

#Plain gradient with adagrad:
Beta3 = plain_gradient_adagrad(learn_rate, n_iterations, X, y, Beta_start)

#Stochastic gradient descent without momentum:
Beta4 = SGD(M, n_epoch, learn_rate, X, y, Beta_start)

#Stochastic gradient descent with momentum:
Beta5 = SGD_momentum(M, n_epoch, learn_rate, X, y, Beta_start)



# for i in range(n_iterations):
#     Beta1 = plain_gradient(learn_rate, i, X, y, Beta_start )
#     Beta2 = plain_gradient_momentum(learn_rate, i, X, y, Beta_start)
#     Beta3 = SGD(M, n_epoch, learn_rate, X, y, Beta_start)



y_pred = X @ Beta3
plt.plot(x, y, "ro")
plt.plot(x, y_pred, "bo")
plt.show()
