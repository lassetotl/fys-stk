"""
The inability for a machine learning method (like linear regression) to
capture the true relationship is called bias.

For example, if we try to apply linear regression on a dataset that clearly behaves as second order poly. the
the bias is relatively large. We then can predict it with a complex line that behaves just like the train-dataset
and see that the mean square error i zero. However, if we no apply this complex model onto the test dataset, we can
often see that it does not match the dataset at all, and see that maybe the linear model fits better (has lower variance).
We call the complex model to be overfitted.

Bias-variance tradeoff deals with finding the sweet spot between these two models, as in our example, finding the
sweet spot between the linear model and complex model.
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.model_selection as skl_datasplit


#Making our own vanilla dataset:

n = 100 #number of datapoints
maxdegree = 5
x = np.linspace(0, 2, n)
y = np.exp(-x**2) + np.random.normal(0, 0.1, n)

#defining arrays for errors, bias and variance between degrees
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
var = np.zeros(maxdegree)


#Splitting between training set and test set:
x_train, x_test, y_train, y_test = skl_datasplit.train_test_split(x, y, test_size=0.2)
