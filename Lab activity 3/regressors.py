from enum import Enum
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import streamlit as st
import pandas as pd
import joblib

# Define model mappings
MODEL_MAPPING = {
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Elastic Net": ElasticNet,
    "AdaBoost Regressor": AdaBoostRegressor,
    "K-Nearest Neighbors": KNeighborsRegressor,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "Linear Regression": LinearRegression,
    "MLP Regressor": MLPRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Support Vector Regressor (SVR)": SVR,
}

# Function to create model with StandardScaler to scale features
def create_model(model_type, **kwargs):
    model_class = MODEL_MAPPING.get(model_type)
    if model_class is None:
        raise ValueError('Invalid model type')

    # Create a pipeline to standardize data and apply the model
    model = make_pipeline(StandardScaler(), model_class(**kwargs))
    st.session_state.reggy_models[model_type] = model
    return model

# Cross-validation and model scoring
def score_model(model, model_type, X, Y, cv):
    results = cross_val_score(model, X, Y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -results.mean()
    mae_std = results.std()
    st.session_state.reggy_scores[model_type] = (mae, mae_std)
    return mae, mae_std

# Ridge Regressor view
def ridge_view():
    alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key='ridgeA')
    max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key='ridgeB')

    model = create_model(
        "Ridge",
        alpha=alpha, max_iter=max_iter
    )
    score_model(model, "Ridge", st.session_state.x, st.session_state.y, st.session_state.cv)

# Linear Regression view
def linear_regression_view():
    st.write("Linear Regression has no hyperparameters to tune.")

    model = create_model("Linear Regression")
    score_model(model, "Linear Regression", st.session_state.x, st.session_state.y, st.session_state.cv)

# Decision Tree Regressor view
def decision_tree_view():
    max_depth = st.slider("Max Depth", 1, 20, None)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)
    n_splits = st.slider("Number of Folds (K)", 2, 20, 10)

    model = create_model(
        "Decision Tree Regressor",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    score_model(model, "Decision Tree Regressor", st.session_state.x, st.session_state.y, st.session_state.cv)

# Elastic Net view
def elastic_net_view():
    alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1)
    l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
    max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

    model = create_model(
        "Elastic Net",
        alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=None
    )
    score_model(model, "Elastic Net", st.session_state.x, st.session_state.y, st.session_state.cv)

# AdaBoost Regressor view
def adaboost_view():
    n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
    learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)

    model = create_model(
        "AdaBoost Regressor",
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=None
    )
    score_model(model, "AdaBoost Regressor", st.session_state.x, st.session_state.y, st.session_state.cv)

# KNN Regressor view
def knn_view():
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
    weights = st.selectbox("Weights", ["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

    model = create_model(
        "K-Nearest Neighbors",
        n_neighbors=n_neighbors, weights=weights, algorithm=algorithm
    )
    score_model(model, "K-Nearest Neighbors", st.session_state.x, st.session_state.y, st.session_state.cv)

# Lasso Regressor view
def lasso_view():
    alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key='lassoA')
    max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key='lassoB')

    model = create_model(
        "Lasso",
        alpha=alpha, max_iter=max_iter, random_state=None
    )
    score_model(model, "Lasso", st.session_state.x, st.session_state.y, st.session_state.cv)

# MLP Regressor view
def mlp_regressor_view():
    hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10)
    activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3)
    solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0)
    learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0)
    max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key="max1")
    random_state = st.number_input("Random State", value=50)

    model = create_model(
        "MLP Regressor",
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state
    )
    score_model(model, "MLP Regressor", st.session_state.x, st.session_state.y, st.session_state.cv)

# Random Forest Regressor view
def random_forest_regressor_view():
    n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
    max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None)
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1)
    random_state = st.number_input("Random State", value=42)

    model = create_model(
        "Random Forest Regressor",
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    score_model(model, "Random Forest Regressor", st.session_state.x, st.session_state.y, st.session_state.cv)

# SVR view
def svr_view():
    kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    model = create_model(
        "Support Vector Regressor (SVR)",
        kernel=kernel,
        C=C,
        epsilon=epsilon
    )
    score_model(model, "Support Vector Regressor (SVR)", st.session_state.x, st.session_state.y, st.session_state.cv)

# Add views to the MODEL_PARAMS mapping
MODEL_PARAMS = {
    "Decision Tree Regressor": decision_tree_view,
    "Elastic Net": elastic_net_view,
    "AdaBoost Regressor": adaboost_view,
    "K-Nearest Neighbors": knn_view,
    "Lasso": lasso_view,
    "Ridge": ridge_view,
    "Linear Regression": linear_regression_view,
    "MLP Regressor": mlp_regressor_view,
    "Random Forest Regressor": random_forest_regressor_view,
    "Support Vector Regressor (SVR)": svr_view,
}

# Graph showing model accuracies
def show_graph():
    st.write("Model MAE Scores")
    s = pd.Series({k: v[0] for k, v in st.session_state.reggy_scores.items()})
    s_std = pd.Series({k: v[1] for k, v in st.session_state.reggy_scores.items()})
    st.bar_chart(s, horizontal=True)
    st.write("Model MAE Standard Deviations")
    st.bar_chart(s_std, horizontal=True)

    best_model = min(st.session_state.reggy_scores, key=st.session_state.reggy_scores.get)
    st.write(f"Best MAE score:\n**{best_model}**: {st.session_state.reggy_scores[best_model][0]}")

    model_saver()

@st.fragment
def model_saver():
    selected_model = st.selectbox("Select a model to save", list(st.session_state.reggy_models.keys()))
    if st.button("Save Model"):
        model = st.session_state.reggy_models[selected_model]
        joblib.dump(model.fit(st.session_state.x, st.session_state.y), f"{selected_model}.joblib")
        st.success(f"Model saved as {selected_model}.joblib")
