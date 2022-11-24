import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

df = pd.read_csv('../heart_cleveland_upload.csv')
print(np.shape(df))
print(f"The shape of the dataset is: {np.shape(df)}")
#print(df.drop('condition', axis=1))
print(df.condition[6])
#target = heat_train('codition')
#print(np.shape(df.condition))
for i in range(40):
    print(df.condition[i])
