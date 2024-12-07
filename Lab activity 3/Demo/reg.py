import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Streamlit app title
st.title("Decision Tree Regressor")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("Decision Tree Regressor Hyperparameters", expanded=True):
    max_depth = st.slider("Max Depth", 1, 20, None)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)
    n_splits = st.slider("Number of Folds (K)", 2, 20, 10)

# Split the dataset into K-Folds
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

# Calculate the mean absolute error using cross-validation
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Display the results in the app
mae = -results.mean()
mae_std = results.std()
st.write(f"Mean Absolute Error (MAE): {mae:.3f} (Standard Deviation: {mae_std:.3f})")

###############################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet

# Streamlit app title
st.title("Elastic Net")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("Elastic Net Hyperparameters", expanded=True):
    alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1)
    l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
    max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the Elastic Net model
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=None)

# Calculate the mean absolute error
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Display the results
st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")

###################################################################################
import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostRegressor

# Streamlit app title
st.title("AdaBoost Regressor")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("AdaBoost Regressor Hyperparameters", expanded=True):
    n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
    learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the AdaBoost model
ada_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=None)

# Calculate the mean absolute error with AdaBoost
scoring = 'neg_mean_absolute_error'
ada_results = cross_val_score(ada_model, X, Y, cv=kfold, scoring=scoring)

# Display the results
st.write(f"AdaBoost Mean Absolute Error (MAE): {-ada_results.mean():.3f} ± {ada_results.std():.3f}")

######################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Streamlit app title
st.title("K-Nearest Neighbors")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=True):
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
    weights = st.selectbox("Weights", ["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the K-NN model
knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

# Calculate the mean absolute error with K-NN
scoring = 'neg_mean_absolute_error'
knn_results = cross_val_score(knn_model, X, Y, cv=kfold, scoring=scoring)

# Display the results
st.write(f"K-NN Mean Absolute Error (MAE): {-knn_results.mean():.3f} ± {knn_results.std():.3f}")

################################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge

# Streamlit app title
st.title("Lasso and Ridge Regression")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Collapsible sidebar for hyperparameters
with st.sidebar.expander("Lasso and Ridge Regression Hyperparameters", expanded=True):
    alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01)
    max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the data on a Lasso Regression model
lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=None)

# Calculate the mean absolute error with Lasso
scoring = 'neg_mean_absolute_error'
lasso_results = cross_val_score(lasso_model, X, Y, cv=kfold, scoring=scoring)

# Display Lasso results
st.write(f"Lasso Mean Absolute Error (MAE): {-lasso_results.mean():.3f} ± {lasso_results.std():.3f}")

# Train the data on a Ridge Regression model
ridge_model = Ridge(alpha=alpha, max_iter=max_iter, random_state=None)

# Calculate the mean absolute error with Ridge
ridge_results = cross_val_score(ridge_model, X, Y, cv=kfold, scoring=scoring)

# Display Ridge results
st.write(f"Ridge Mean Absolute Error (MAE): {-ridge_results.mean():.3f} ± {ridge_results.std():.3f}")

###########################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Streamlit app title
st.title("Linear Regression")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Collapsible sidebar for additional options if needed
with st.sidebar.expander("Linear Regression Hyperparameters", expanded=True):
    # Placeholder for future options, if needed
    st.write("N/A")

# Train the data on a Linear Regression model
model = LinearRegression()

# Calculate the mean absolute error
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Display results
st.write(f"Mean Absolute Error (MAE): {-results.mean():.3f} ± {results.std():.3f}")

############################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor

# Streamlit app title
st.title("MLP Regressor")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Create a collapsible sidebar for hyperparameters
with st.sidebar.expander("MLP Regressor Hyperparameters", expanded=True):
    # Define hyperparameters with default values
    hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10)
    activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3)
    solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0)
    learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0)
    max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key="max1")
    random_state = st.number_input("Random State", value=50)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the data on an MLP Regressor with specified hyperparameters
mlp_model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,  # Use user-defined hidden layer sizes
    activation=activation,                   # Use user-defined activation function
    solver=solver,                          # Use user-defined optimization algorithm
    learning_rate=learning_rate,            # Use user-defined learning rate schedule
    max_iter=max_iter,                      # Use user-defined max iterations
    random_state=random_state                # Use user-defined random state
)

# Calculate the mean absolute error with MLP
scoring = 'neg_mean_absolute_error'
mlp_results = cross_val_score(mlp_model, X, Y, cv=kfold, scoring=scoring)

# Display results
st.write("Mean Absolute Error (MAE): %.3f ± %.3f" % (-mlp_results.mean(), mlp_results.std()))

##################################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Streamlit app title
st.title("Random Forest Regressor")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Create a collapsible sidebar for hyperparameters
with st.sidebar.expander("Random Forest Regressor Hyperparameters", expanded=True):
    n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
    max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
    random_state = st.number_input("Random State", value=42)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the data on a Random Forest Regressor with specified hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=n_estimators,      # Use user-defined number of trees
    max_depth=max_depth,            # Use user-defined max depth
    min_samples_split=min_samples_split,  # Use user-defined min samples split
    min_samples_leaf=min_samples_leaf,    # Use user-defined min samples leaf
    random_state=random_state        # Use user-defined random state
)

# Calculate the mean absolute error with Random Forest
scoring = 'neg_mean_absolute_error'
rf_results = cross_val_score(rf_model, X, Y, cv=kfold, scoring=scoring)

# Display results
st.write("Random Forest Mean Absolute Error (MAE): %.3f ± %.3f" % (-rf_results.mean(), rf_results.std()))

###########################################################################

import streamlit as st
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

# Streamlit app title
st.title("Support Vector Regressor (SVR)")

# Load the dataset
filename = './housing.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Create a collapsible sidebar for hyperparameters
with st.sidebar.expander("Support Vector Regressor (SVR) Hyperparameters", expanded=True):
    kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

# Split the dataset into a 10-fold cross-validation
kfold = KFold(n_splits=10, random_state=None)

# Train the data on a Support Vector Regressor with specified hyperparameters
svm_model = SVR(
    kernel=kernel,          # Use user-defined kernel
    C=C,                    # Use user-defined regularization parameter
    epsilon=epsilon         # Use user-defined epsilon
)

# Calculate the mean absolute error with SVM
scoring = 'neg_mean_absolute_error'
svm_results = cross_val_score(svm_model, X, Y, cv=kfold, scoring=scoring)

# Display results
st.write("SVM Mean Absolute Error (MAE): %.3f ± %.3f" % (-svm_results.mean(), svm_results.std()))
