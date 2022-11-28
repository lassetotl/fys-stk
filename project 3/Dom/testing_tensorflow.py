import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

from sklearn.model_selection import train_test_split

from sklearn import datasets

import warnings

#suppress warnings
warnings.filterwarnings('ignore')

np.random.seed(0)

# # download MNIST dataset
# digits = datasets.load_digits()
#
# # define inputs and labels
# inputs = digits.images
# labels = digits.target
# n_inputs = len(inputs)
# inputs = inputs.reshape(n_inputs, -1)
#
# labels = to_categorical(labels)
# train_size = 0.8
# test_size = 1 - train_size
# X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
#                                                     test_size=test_size)

data = '../data/Preprocessed_heart_data.npy'
X_train, X_test, Y_train, Y_test = np.load(data, allow_pickle=True)
Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
Y_train=to_categorical(Y_train)     #Convert labels to categorical when using categorical cross entropy
Y_test=to_categorical(Y_test)




epochs = 300
batch_size = 100
n_neurons_layer1 = 300
n_neurons_layer2 = 200
n_neurons_layer3 = 200
n_neurons_layer4 = 200
n_neurons_layer5 = 200
n_neurons_layer6 = 200
n_neurons_layer7 = 200

n_categories = 2
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta, lmbd):
    #print(n_neurons_layer3)
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer3, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer4, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer5, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer6, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer6, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer6, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer6, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))

    model.add(Dense(2, activation='sigmoid'))

    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model

DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta=eta, lmbd=lmbd)

        DNN.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        scores = DNN.evaluate(X_train, Y_train)

        DNN_keras[i][j] = DNN
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print("Test accuracy: %.3f" % scores[1])
        print()
