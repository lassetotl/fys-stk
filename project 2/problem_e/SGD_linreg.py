import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import seaborn as sns
from SGD_train import *
from sigmoid import *
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Defining the data we are using, and splitting it
data, target = load_breast_cancer(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

#Defining the function for analysis
def analysis(X_test, X_train, y_test, y_train, data):
    learning_rate = np.logspace(-3, 0, 10)
    lmbd = np.logspace(-3, 0, 10)
    learning_rate = learning_rate.round(decimals=3)
    lmbd = lmbd.round(decimals=3)

    M = 10 #Batch size
    n_epochs = 70 #number of epochs
    features = X_train.shape[1]
    w = np.zeros([features, 1])
    b = 0
    lmbd_ = 0.1
    learning_rate_ = 0.01

    train_check = np.zeros([len(learning_rate), len(lmbd)])
    test_check = np.zeros([len(learning_rate), len(lmbd)])

    for i, eta in enumerate(learning_rate):
        for j, lmd in enumerate(lmbd):
            w, b = SGD_train(M, n_epochs, X_train, y_train, w, b, lmbd_, learning_rate_)

            pred_test = sigmoid(X_test @ w + b)
            pred_train = sigmoid(X_train @ w + b)
            pred_test[pred_test<0.5] = 0
            pred_test[pred_test>=0.5] = 1
            pred_train[pred_train<0.5] = 0
            pred_train[pred_train>=0.5] = 1

            train_check[i][j] = accuracy_score(y_train.reshape(pred_train.shape), pred_train)
            test_check[i][j] = accuracy_score(y_test.reshape(pred_test.shape), pred_test)

    fig, axis = plt.subplots(2)
    sns.heatmap(test_check, xticklabels=learning_rate, yticklabels=lmbd, annot=True, ax=axis[0], cmap="viridis")
    plt.ylabel("lambda")
    plt.xlabel("learning rate")
    sns.heatmap(train_check, xticklabels=learning_rate, yticklabels=lmbd, annot=True, ax=axis[1], cmap="viridis")
    plt.ylabel("lambda")
    plt.xlabel("learning rate")
    axis[0].set_title("Test dataset")
    axis[1].set_title("Train dataset")
    plt.show()





analysis(test_data, train_data, test_target, train_target, data)
