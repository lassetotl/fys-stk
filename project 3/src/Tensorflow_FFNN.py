#Imports:
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

#Ignore overflow warnings:
import warnings
warnings.filterwarnings('ignore')


#Importing the data
data = '../data/All_heart_data.npy'
data_train, data_test, target_train, target_test = np.load(data, allow_pickle=True)

#Converting the target vectors to binary class matrix
target_train = to_categorical(target_train)
target_test = to_categorical(target_test)

#variables and hyperparameters
epochs = 300
#batch_size = 100
batch_size = np.linspace(10, 200, 20,dtype='int64')
#n_neurons = 20
n_neurons = np.linspace(5, 80, 16,dtype=np.int64)
n_categories = 2
#n_layers = 1
n_layers = np.linspace(1,10,10,dtype=np.int64)



eta = 0.01
lmbd = 0.01

def create_NN(n_neurons, n_categories, n_layers, eta, lmbd):
    model = Sequential() #initiating the network model
    model.add(Dense(n_neurons,activation='relu',kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

def create_NN_search(n_neurons, n_categories, n_layers, eta, lmbd):
    for i, a in enumerate(n_layers):
        for j, b in enumerate(batch_size):
            for k, c in enumerate(n_neurons):
                model = Sequential() #initiating the network model
                for layers in range(a): #adding the hidden layers
                    model.add(Dense(c,activation='relu',kernel_regularizer=regularizers.l2(lmbd)))
                model.add(Dense(n_categories, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                model.fit(data_train, target_train, epochs=epochs, batch_size=b, verbose=1)
                scores = model.evaluate(data_test, target_test)

                analyse_file = open("analyse.txt","a")
                analyse_file.write(f"n_layers: {a}, batch_size:{b}, n_neurons:{c} gives score of: {scores} \n")
    return model



model1 = create_NN(n_neurons=90, n_categories=2, n_layers=1, eta=0.00001, lmbd=0.01)
model1.fit(data_train, target_train, epochs=2000, batch_size=70, verbose=1)
scores1 = model1.evaluate(data_test, target_test)
print(scores1)

model2 = create_NN_search(n_neurons=90, n_categories=2, n_layers=1, eta=0.00001, lmbd=0.01)
model1.fit(data_train, target_train, epochs=2000, batch_size=70, verbose=1)
scores1 = model1.evaluate(data_test, target_test)
print(scores1)
