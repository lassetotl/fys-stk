## Data used in project 3

### Heart_cleveland_upload.csv

File contains the dataset gathered in Cleveland specifically, with categorical feature values converted to numerical. Condition given in binary. 

### Heart_disease_uci.csv

Modified version containing all datasets from Cleveland, Hungary, Switzerland and VA Long Beach. Categorical values as strings, condition as multiclass (0,1,2,3,4).

### Preprocessed_heart_data.npy

Our own preprocessing of the data in 'Heart_cleveland_upload.csv', after feature extraction and a train/test split with 20% of the instances allocated to the test sets. When loading dataset in python, call four variables like this (with numpy as np): \
\
X_train, X_test, y_train, y_test = np.load('Preprocessed_heart_data.npy', allow_pickle=True). 
