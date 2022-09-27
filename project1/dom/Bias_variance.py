#Imports

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# n = 500
# n_boostraps = 100
# degree = 18
# noise = 0.1
#
# #Making the data
# x = np.linspace(-1, 3, n).reshape(-1,1)
# y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
#
# #splitting data into test data and train data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# #Making a matrix that will hold the column vectors as y_pred
# y_pred = np.empty([y_test.shape[0], n_boostraps])
#
# model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False)) #Still no idea what this does
#
# for i in range(n_boostraps):
#     x_, y_ = resample(x_train, y_train)
#     y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
#
# error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
# bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
# variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
# print('Error:', error)
# print('Bias^2:', bias)
# print('Var:', variance)
# print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

#####################################################################################################
# np.random.seed(2018)
#
# n = 40
# n_boostraps = 100
# maxdegree = 14
#
#
# # Make data set.
# x = np.linspace(-3, 3, n).reshape(-1, 1)
# y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
# error = np.zeros(maxdegree)
# bias = np.zeros(maxdegree)
# variance = np.zeros(maxdegree)
# polydegree = np.zeros(maxdegree)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# for degree in range(maxdegree):
#     model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
#     y_pred = np.empty((y_test.shape[0], n_boostraps))
#     for i in range(n_boostraps):
#         x_, y_ = resample(x_train, y_train)
#         y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
#
#     polydegree[degree] = degree
#     error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
#     bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
#     variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
#     print('Polynomial degree:', degree)
#     print('Error:', error[degree])
#     print('Bias^2:', bias[degree])
#     print('Var:', variance[degree])
#     print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
#
# plt.plot(polydegree, error, label='Error')
# plt.plot(polydegree, bias, label='bias')
# plt.plot(polydegree, variance, label='Variance')
# plt.legend()
# plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error


degrees = 50
testerror = np.zeros(degrees)
trainingerror = np.zeros(degrees)
polynomial = np.zeros(degrees)


trials = 300
for polydegree in range(1, degrees):
    polynomial[polydegree] = polydegree
    testerror[polydegree] = 0.0
    trainingerror[polydegree] = 0.0
 
    for samples in range(trials):
        x_train, x_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=0.2)
        model = LinearRegression(fit_intercept=False).fit(x_train, z_train)
        zpred = model.predict(x_train)
        ztilde = model.predict(x_test)
        testerror[polydegree] += mean_squared_error(z_test, ztilde)
        trainingerror[polydegree] += mean_squared_error(z_train, zpred)

    testerror[polydegree] /= trials
    trainingerror[polydegree] /= trials
    print("Degree of polynomial: %3d"% polynomial[polydegree])
    print("Mean squared error on training data: %.8f" % trainingerror[polydegree])
    print("Mean squared error on test data: %.8f" % testerror[polydegree])

plt.plot(polynomial, np.log10(trainingerror), label='Training Error')
plt.plot(polynomial, np.log10(testerror), label='Test Error')
plt.xlabel('Polynomial degree')
plt.ylabel('log10[MSE]')
plt.legend()
plt.show()
