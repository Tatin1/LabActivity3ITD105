import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Streamlit app title
st.title("Decision Tree ML")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("Decision Tree Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
    random_seed = st.slider("Random Seed", 1, 100, 50)
    max_depth = st.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Initialize and train the Decision Tree Classifier with hyperparameters
model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=random_seed
)

model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")


###################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Streamlit app title
st.title("Gaussian Naive Bayes")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters with unique keys
with st.sidebar.expander("Gaussian Naive Bayes Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
    var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="var_smoothing")

# Convert var_smoothing from log scale to regular scale
var_smoothing_value = 10 ** var_smoothing

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Initialize the Gaussian Naive Bayes classifier with hyperparameters
model = GaussianNB(var_smoothing=var_smoothing_value)

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

###############################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

# Streamlit app title
st.title("AdaBoost")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("AdaBoost Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size2")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed2")
    n_estimators = st.slider("Number of Estimators", 1, 100, 50)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create an AdaBoost classifier
model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

###############################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Streamlit app title
st.title("K-Nearest Neighbors")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="keytest3")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed3")
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
    weights = st.selectbox("Weights", options=["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create a K-Nearest Neighbors (K-NN) classifier
model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

###############################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Streamlit app title
st.title("Logistic Regression")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("Logistic Regression Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
    max_iter = st.slider("Max Iterations", 100, 500, 200)
    solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
    C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)

# Train the model on the training data
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

######################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Streamlit app title
st.title("MLP Classifier")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("MMLP Classifier Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
    activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")

# Convert hidden_layer_sizes input to tuple
hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create an MLP-based model
model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                      solver='adam', max_iter=max_iter, random_state=random_seed)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

######################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Streamlit app title
st.title("Perceptron Classifier")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("Perceptron Classifier Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test6")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed6")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max6")
    eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
    tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create a Perceptron classifier
model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = model.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

##########################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Streamlit app title
st.title("Random Forest")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Set collapsible sidebar parameters for hyperparameters
with st.sidebar.expander("Random Forest Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
    n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
    max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
    min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create a Random Forest classifier
rfmodel = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_seed,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf
)

# Train the model
rfmodel.fit(X_train, Y_train)

# Evaluate the accuracy
accuracy = rfmodel.score(X_test, Y_test)

# Display the accuracy in the app
st.write(f"Accuracy: {accuracy * 100.0:.3f}%")

#################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Streamlit app title
st.title("Support Vector Machine (SVM)")

# Load the dataset
filename = './pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("Support Vector Machine (SVM) Hyperparameters", expanded=True):
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
    random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

# Create an SVM classifier
model = SVC(kernel=kernel, C=C, random_state=random_seed)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the accuracy
result = model.score(X_test, Y_test)
st.write(f"Accuracy: {result * 100.0:.3f}%")

