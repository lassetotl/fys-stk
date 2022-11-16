import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LogisticRegression

np.random.seed(20)

#getting and setting up the data
digits = datasets.load_digits()
inputs = digits.images
labels = digits.target

n = len(inputs)
print(f"the number of inputs are: {n}")

#flatting out the image
inputs = inputs.reshape(n, -1)

# choose some random images to display
indices = np.arange(n)
random_indices = np.random.choice(indices, size=5)

def plot_digits():
    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
        plt.show()

#We wish to split the data into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#Now we scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

def append_intercept(X):
    xb = np.ones((X.shape[0], 1))
    return np.c_[xb, X]

X_train = append_intercept(X_train)
X_test = append_intercept(X_test)
X_val = append_intercept(X_val)


print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))


# #The logistic accuracy
# learning_rate = 0.01
# n_categories = np.max(labels) + 1
