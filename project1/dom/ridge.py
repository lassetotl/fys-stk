import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#from lecture notes on Ridge/Lasso regression
def create_X(x, y, n): #design matrices for polynomials up to 10th degree
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    #print(f'Features/Length beta: {l}') #what amount should we expect?
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X


# Making meshgrid of datapoints and compute Franke's function
n = 5 #in task b; up to 5 (this affects the approximate terrain plot)
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

x_, y_ = np.meshgrid(x,y)

#how does the scaling affect the experiment?
var = 0.1 #variance of noise (?)
noise = np.random.normal(0, var, len(x_)*len(x_))
noise = noise.reshape(len(x_), len(x_))

z = FrankeFunction(x_, y_) + noise #Franke with noise

X = create_X(x_, y_, n)

# We split the data in test and training data, 20%
X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=0.2)



I = np.eye(np.shape(X_train)[1],np.shape(X_train)[1])

nlambdas = 100
MSEPredict = np.zeros(nlambdas)
lambdas = np.logspace(-10,10, nlambdas)




for i in range(nlambdas):
    lmb = lambdas[i]
    RidgeBeta = np.linalg.inv(X_train.T@X_train + lmb*I) @ X_train.T @ z_train
    zpredictRidge = X_train @ RidgeBeta
    MSEPredict[i] = MSE(z_train,zpredictRidge)

plt.figure()
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Train')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
#plt.show()

# lamb_opt = 0.397
# RidgeBeta = np.linalg.inv(X.T@X + 0.397*I) @ X.T @ z
# zpredictRidge = X_train @ RidgeBeta
#
# zmatrix = zpredictRidge.reshape(100,100)
#
#
# print(np.size(zpredictRidge))
# print(np.shape(zpre))
#
#
# zmatrix = linreg.predict(X_train)
# zmatrix = zmatrix.reshape(100,100)


# fig = plt.figure(figsize = (13, 7))
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(x_, y_, zpredictRidge, cmap = cm.coolwarm,
# linewidth = 0, antialiased = False)
#
# #predict_surf = ax.plot_trisurf(x, y, zpredict, cmap = cm.coolwarm,
# #linewidth = 0, antialiased = False)
# # Customize the z axis.
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title(f'OLS approximation at order n = {n}', fontsize = 15)
# plt.show()
