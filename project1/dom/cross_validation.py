import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

n = 100
x = np.random.randn(n)
y = 3*x**2 + np.random.randn(n)

poly = PolynomialFeatures(degree=6)

k = 5
kfold = KFold(n_splits = k)

scores_KFold = np.zeros([nla])
