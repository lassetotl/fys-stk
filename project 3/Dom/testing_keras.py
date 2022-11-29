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


data = '../data/Preprocessed_heart_data.npy'
data_all = '../data/All_heart_data.npy'
X_train, X_test, Y_train, Y_test = np.load(data, allow_pickle=True)
# Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
# Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))

Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

eta = np.logspace(-3, 1, 3,dtype=int)
lmbd = 0.01
n_layers = 2
n_neuron = np.logspace(0,3,4,dtype=int)
epochs = 100
batch_size = 100

def NN_model(inputsize, n_layers, n_neurons, eta):
    model = Sequential()
    for i in range(n_layers):
        if (i==0):
            model.add(Dense(n_neurons,activation='relu',kernel_regularizer=regularizers.l2(0.01),input_dim=inputsize))
        else:
            model.add(Dense(n_neurons,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

Train_accuracy=np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for

for i in range(len(n_neuron)):
    for j in range(len(eta)):
        DNN_model = NN_model(X_train.shape[1], n_layers, n_neuron[i], eta[j])
        DNN_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, )
        print(f"neurons: {n_neuron[i]}")
        print(f"eta: {eta[j]}")
        Train_accuracy[i,j]=DNN_model.evaluate(X_train,Y_train)[1]
        Test_accuracy[i,j]=DNN_model.evaluate(X_test,Y_test)[1]
plot_data(eta,n_neuron,Train_accuracy, 'training')
plot_data(eta,n_neuron,Test_accuracy, 'testing')
plt.show()
