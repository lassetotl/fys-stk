from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from SGD_train import *
from sigmoid import *

#Defining the data we are using, and splitting it
data, target = load_breast_cancer(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

logreg = LogisticRegression(solver='lbfgs',
                                multi_class='multinomial',
                                penalty='l2',
                                max_iter=100,
                                random_state=42,
                                C=1e5)

logreg.fit(train_data, train_target)
predict_sklearn = logreg.predict(test_data)

print('scikit learn: {}'.format(sum(predict_sklearn == test_target)/(float(len(test_target)))))
