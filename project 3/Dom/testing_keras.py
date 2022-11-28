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
X_train, X_test, Y_train, Y_test = np.load(data, allow_pickle=True)
# Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
# Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))

Y_train=to_categorical(Y_train)     #Convert labels to categorical when using categorical cross entropy
Y_test=to_categorical(Y_test)
