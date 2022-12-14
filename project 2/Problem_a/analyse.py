import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns


from data import *
from plain_gradient import *
from plain_gradient_momentum import *
from SGD import *
from SGD_momentum import *
from plain_gradient_adagrad import *
from plain_gradient_adagrad_momentum import *
from SGD_adagrad import *
from SGD_adagrad_momentum import *
from SGD_RMSprop import *
from SGD_ADAM import *

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

n = 100 #number of datapoints
x, y, X, max_EigVal = DataPoints(n).output()


#Variables:
learn_rate = 1/max_EigVal
n_iterations = 70
Beta_start = np.random.randn(np.shape(X)[1],1)
M = 10 #Size of each minibatch
n_epoch = 70 #number of epochs

"""
Doing some heatmap
"""
def heatmap():
    learning_rate = np.logspace(-3, 0, 10)
    lmbd = np.logspace(-3, 0, 10)
    learning_rate = learning_rate.round(decimals=3)
    lmbd = lmbd.round(decimals=3)

    check = np.zeros([len(learning_rate), len(lmbd)])
    for i, eta in enumerate(learning_rate):
        for j, lmd in enumerate(lmbd):
            Beta = SGD_momentum(M, n_epoch, eta, X, y, Beta_start, lmd)
            y_pred = X @ Beta

            check[i][j] = MSE(y, y_pred)
    check = check/(np.max(check))
    df = pd.DataFrame(data=check)
    sns.heatmap(df, xticklabels=learning_rate, yticklabels=lmbd, annot=True, fmt='.2g')
    plt.ylabel("lambdas")
    plt.xlabel("learning rates")
    plt.show()

"""
Analysis of plain gradient descents:
"""
def plain_GD_analysis():
    N = np.zeros(n_iterations)
    MSE1 = np.zeros(n_iterations)
    MSE2 = np.zeros(n_iterations)
    MSE3 = np.zeros(n_iterations)
    MSE4 = np.zeros(n_iterations)
    for i in range(n_iterations):
        Beta1 = plain_gradient(learn_rate, i, X, y, Beta_start,0 )
        y_pred1 = X @ Beta1
        MSE1[i] = MSE(y,y_pred1)
        Beta2 = plain_gradient_momentum(learn_rate, i, X, y, Beta_start,0)
        y_pred2 = X @ Beta2
        MSE2[i] = MSE(y,y_pred2)
        Beta3 = plain_gradient_adagrad(learn_rate, i, X, y, Beta_start,0)
        y_pred3 = X @ Beta3
        MSE3[i] = MSE(y,y_pred3)
        Beta4 = plain_gradient_adagrad_momentum(learn_rate, i, X, y, Beta_start,0)
        y_pred4 = X @ Beta4
        MSE4[i] = MSE(y,y_pred4)




        N[i] = i

    print(f"minimum MSE in plain gradient is: {min(MSE1)}")
    print(f"minimum MSE in plain gradient with momentum is: {min(MSE2)}")
    print(f"minimum MSE in plain gradient with adagrad is: {min(MSE3)}")
    print(f"minimum MSE in plain gradient with adagrad and momentum is: {min(MSE4)}")

    plt.plot(N, MSE1, label="plain gradient")
    plt.plot(N, MSE2, label="plain gradient with momentum")
    plt.plot(N, MSE3, label="plain gradient with adagrad")
    plt.plot(N, MSE4, label="plain gradient with adagrad and momentum")
    plt.grid(True)
    plt.xlabel("number of iterations")
    plt.ylabel("MSE")
    plt.title("Plain gradient descent with extensions")
    plt.legend()
    plt.show()
"""
Analysis of stochastic gradient descents
"""
def SGD_analyse():
    N = np.zeros(n_epoch)
    MSE1 = np.zeros(n_epoch)
    MSE2 = np.zeros(n_epoch)
    MSE3 = np.zeros(n_epoch)
    MSE4 = np.zeros(n_epoch)
    MSE5 = np.zeros(n_epoch)
    MSE6 = np.zeros(n_epoch)

    for i in range(n_epoch):
        Beta1 = SGD(M, i, learn_rate, X, y, Beta_start,0)
        y_pred1 = X @ Beta1
        MSE1[i] = MSE(y,y_pred1)

        Beta2 = SGD_momentum(M, i, learn_rate, X, y, Beta_start,0)
        y_pred2 = X @ Beta2
        MSE2[i] = MSE(y,y_pred2)

        Beta3 = SGD_adagrad(M, i, learn_rate, X, y, Beta_start,0)
        y_pred3 = X @ Beta3
        MSE3[i] = MSE(y,y_pred3)

        Beta4 = SGD_adagrad_momentum(M, i, learn_rate, X, y, Beta_start,0)
        y_pred4 = X @ Beta4
        MSE4[i] = MSE(y,y_pred4)

        Beta5 = SGD_RMSprop(M, i, learn_rate, X, y, Beta_start,0)
        y_pred5 = X @ Beta5
        MSE5[i] = MSE(y,y_pred5)

        Beta6 = SGD_ADAM(M, i, learn_rate, X, y, Beta_start,0)
        y_pred6 = X @ Beta6
        MSE6[i] = MSE(y,y_pred6)


        N[i] = i

    print(f"minimum MSE in SGD is: {min(MSE1)}")
    print(f"minimum MSE SGD with momentum is: {min(MSE2)}")
    print(f"minimum MSE in SGD with adagrad is: {min(MSE3)}")
    print(f"minimum MSE in SGD with adagrad and momentum is: {min(MSE4)}")

    plt.plot(N, MSE1, label="simple SDG")
    plt.plot(N, MSE2, label="simple SDG with momentum")
    plt.plot(N, MSE3, label="simple SDG with adagrad")
    plt.plot(N, MSE4, label="simple SDG with adagrad and momentum")
    #plt.plot(N, MSE5, label="simple SDG with RMSProp method")
    #plt.plot(N, MSE6, label="simple SDG with ADAM method")
    plt.grid(True)
    plt.xlabel("number of epochs")
    plt.ylabel("MSE")
    plt.title("Stochastic gradient descent with extensions")
    plt.legend()
    plt.show()


plain_GD_analysis()
SGD_analyse()
heatmap()
