The third project in Applied Data Analysis and Machine Learning by Lasse Totland, Domantas Sakalys, Synne Mo Sandnes, and Semya A. Tønnessen. 

## Abstract 
This report covers the development of two binary classification machine learning models for prediction of heart disease diagnosis using the 1988 Cleveland Clinic Foundation survey data on heart disease among hospital patients. The dataset contains 14 attributes relevant to our analysis and we determined that none of them should be extracted during preprocessing as all of them contributed positively on the prediction test accuracy of our models. Using a classification Feed Forward Neural Network with back-propagation (and activation functions RELU for one hidden layer and softmax for output layer) we achieved an accuracy of $0.91$ with learning rate $\eta = 0.01$ and regularization factor $\lambda = 0.001$ using hyperparameter analysis. Using a Random Forest Classifier with Adaptive Boosting (500 estimators) we achieved a test accuracy of $0.88$ with $\eta = 0.1$ and max tree depth of $1$ using hyperparameter analysis, as well as a $0.89$ true positive prediction accuracy. These results are comparable with established literature analysing the same dataset.

## Dependencies 
Requires python3 
Check your version by running 
``` python3 --version ``` 

The project makes use of the following libraries: 
- `numpy`
- `matplotlib`
- `time`
- `seaborn`
- `pandas`
- `sklearn`
- `keras`
- `random`
- `tensorflow`

## Running the code 
The output from the main project is generated in the `.ipynb` files. 

The output from the bonus task (bias-variance analysis) can be run from the `src` folder in your terminal using: 
- `python3 bonus/biasvar_analysis.py`
Note that the program takes a bit of time to run. 

## Contents 
### analyse_results 
Folder containing some output. 

### data
Folder that contains various files that are called in the notebooks and python scripts.

### doc
Folder containing the `.pdf` and `.tex` files with the project report, for both project 3 and the bonus assignment. 

### figures 
Folder containing results produced, as well as a subfolder `bonus` containing the figures for the bias-variance analysis. 

### src 
Folder that contains the codes used to produce results, as well as a subfolder `bonus` containing the codes for the bias-variance analysis. 

- `Building_own_FFNN.py` : Python Class which is our own writte backbone for Feed forwards neural network. This class has function for encoding for onehot vectors, feed forward function, backpropagation and training function for our network. It also includes function for sigmoid function and a function that calculated the predictions. 
- `Dataprep_A.ipynb`: Jupyter notebook that covers our data preparation with pandas, incl. correlation analysis and feature extraction.
- `Random_Forest_Analysis.ipynb`: Notebook displaying the development of a boosted random forest model for predicting heart disease diagnosis using scikit learn functionalities.
- `Tensorflow_FFNN.py` :  Python script that imports tensorflow library and performs feed forward neural network. There are two function, first function "create_NN" simply performs neural network, while the second function "create_NN_search" performs a grid seatch for optimal architecture.
- `imports_from_ownFFNN.py` : Python script which is used to import from "Building_own_FFNN.py" as does a grid search for finding optimal hyper-parameters.
- `bonus/biasvar_analysis`: Python scrips for the bias-variance analysis for OLS, Ridge, FFNN, Decision Trees, and Random Forests. 

## Tests 
No tests have been made, but it was last ran with the operating system `Ubuntu 22.04 LTS`. 
