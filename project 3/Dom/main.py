import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from Building_NN import Build
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

data = '../data/Preprocessed_heart_data.npy'
data_all = '../data/All_heart_data.npy'
data_train, data_test, target_train, target_test = np.load(data_all, allow_pickle=True)

# target_train1 = np.zeros((237))
# for i in range(len(target_train)):
#     value = target_train[i]
#     target_train1[i] = int(value)



# inst = Build(data_train, target_train, eta = 0.00001, lmbd = 0.00001)
# hw, hb, ow, ob = inst.train()
# probabilities = inst.feed_forward(hw, hb, ow, ob, data_train)
#
# predictions = inst.results(probabilities)
#
# print(np.shape(target_train))
# print(np.shape(predictions))
# for i in range(1000):
#     print(target_train[i], predictions[i])
#

eta_vals = np.logspace(-5, 0.1, 7)
lmbd_vals = np.logspace(-5, 0.1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):

        inst = Build(data_train, target_train, eta=eta, lmbd=lmbd)
        hw, hb, ow, ob = inst.train()
        probabilities = inst.feed_forward(hw, hb, ow, ob, data_train)
        predictions = inst.results(probabilities)

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", accuracy_score(target_train, predictions))
