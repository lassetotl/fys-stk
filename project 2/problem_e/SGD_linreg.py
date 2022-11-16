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

def analysis(X_test, X_train, y_test, y_train, data):
    learning_rate = np.logspace(-3, 0, 10)
    lmbd = np.logspace(-3, 0, 10)

    M = 10 #Batch size
    n_epochs = 50 #number of epochs
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
    return train_check, test_check




train_check, test_check = analysis(test_data, train_data, test_target, train_target, data)

sns.heatmap(test_check, annot=True, cmap="viridis")
plt.xlabel("log(lambda)")
plt.ylabel("log(learning rate)")
plt.show()
sns.heatmap(train_check, annot=True, cmap="viridis")
plt.xlabel("log(lambda)")
plt.ylabel("log(learning rate)")
plt.show()
