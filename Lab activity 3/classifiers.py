from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import streamlit as st
import pandas as pd
import joblib


MODEL_MAPPING = {
    "Decision Tree": DecisionTreeClassifier,
    "Gaussian Naive Bayes": GaussianNB,
    "AdaBoost": AdaBoostClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Logistic Regression": LogisticRegression,
    "MLP Classifier": MLPClassifier,
    "Perceptron Classifier": Perceptron,
    "Random Forest": RandomForestClassifier,
    "Support Vector Machine (SVM)": SVC,
}

def create_model(model_type, **kwargs):
    model_class = MODEL_MAPPING.get(model_type)
    if model_class is None:
        raise ValueError('Invalid model type')
    
    # Create and store the model in session state
    model = model_class(**kwargs)
    st.session_state.classy_models[model_type] = model
    return model

def score_model(model, model_type, X, Y, cv):
    score = cross_val_score(model, X, Y, cv=cv, scoring='accuracy').mean()
    # Store the score in session state
    st.session_state.classy_scores[model_type] = score
    return score

def decision_tree_view():
    with st.container():
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.slider("Random Seed", 1, 100, 50)
        max_depth = st.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    # st.session_state.
    model = create_model(
            "Decision Tree", 
            random_state=random_seed, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf
        )
    score_model(model, "Decision Tree", st.session_state.x, st.session_state.y, st.session_state.cv)

def gaussian_nb_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")
    var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="var_smoothing")
    model = create_model("Gaussian Naive Bayes", var_smoothing=10 ** var_smoothing)
    score_model(model,"Gaussian Naive Bayes", st.session_state.x, st.session_state.y, st.session_state.cv)

def adaboost_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test_size2")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed2")
    n_estimators = st.slider("Number of Estimators", 1, 100, 50)
    model = create_model("AdaBoost", n_estimators=n_estimators, random_state=random_seed)
    score_model(model,"AdaBoost", st.session_state.x, st.session_state.y, st.session_state.cv )

def knn_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="keytest3")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed3")
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
    weights = st.selectbox("Weights", options=["uniform", "distance"])
    algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"])
    model = create_model("K-Nearest Neighbors", n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    score_model(model,"K-Nearest Neighbors", st.session_state.x, st.session_state.y, st.session_state.cv)

def logistic_regression_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test4")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed4")
    max_iter = st.slider("Max Iterations", 100, 500, 200)
    solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
    C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0)
    model = create_model("Logistic Regression", max_iter=max_iter, solver=solver, C=C)
    score_model(model,"Logistic Regression", st.session_state.x, st.session_state.y, st.session_state.cv)

def mlp_classifier_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test5")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed5")
    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", "65,32")
    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
    activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"])
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max5")
    model = create_model("MLP Classifier", hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver='adam', max_iter=max_iter, random_state=random_seed)
    score_model(model, "MLP Classifier", st.session_state.x, st.session_state.y, st.session_state.cv)

def perceptron_classifier_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test6")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed6")
    max_iter = st.slider("Max Iterations", 100, 500, 200, key="max6")
    eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0)
    tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3)
    model = create_model("Perceptron Classifier", max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
    score_model(model, "Perceptron Classifier", st.session_state.x, st.session_state.y, st.session_state.cv)

def random_forest_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test7")
    random_seed = st.slider("Random Seed", 1, 100, 7, key="seed7")
    n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100)
    max_depth = st.slider("Max Depth of Trees", 1, 50, None)  # Allows None for no limit
    min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2)
    min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1)
    model = create_model("Random Forest", n_estimators=n_estimators,
        random_state=random_seed,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf 
        )
    score_model(model, "Random Forest", st.session_state.x, st.session_state.y, st.session_state.cv)

def svm_view():
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="test8")
    random_seed = st.slider("Random Seed", 1, 100, 42, key="seed8")
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
    kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'])
    model = create_model("Support Vector Machine (SVM)",kernel=kernel, C=C, random_state=random_seed)
    score_model(model, "Support Vector Machine (SVM)", st.session_state.x, st.session_state.y, st.session_state.cv)

MODEL_PARAMS = {
    "Decision Tree": decision_tree_view,
    "Gaussian Naive Bayes": gaussian_nb_view,
    "AdaBoost": adaboost_view,
    "K-Nearest Neighbors": knn_view,
    "Logistic Regression": logistic_regression_view,
    "MLP Classifier": mlp_classifier_view,
    "Perceptron Classifier": perceptron_classifier_view,
    "Random Forest": random_forest_view,
    "Support Vector Machine (SVM)": svm_view,
}

def show_graph():
    st.write("Model Accuracies")
    s = pd.Series(st.session_state.classy_scores)
    s.rename("Accuracy", inplace=True)
    st.bar_chart(s, horizontal = True)
    best_model = max(st.session_state.classy_scores, key=st.session_state.classy_scores.get)
    st.write(f"Best model: {best_model}: {st.session_state.classy_scores[best_model]:.2f}")

    model_saver()

@st.fragment
def model_saver():
    selected_model = st.selectbox("Select model to save", list(st.session_state.classy_models.keys()))
    if st.button("Save Model"):
        model = st.session_state.classy_models[selected_model]
        joblib.dump(model.fit(st.session_state.x, st.session_state.y), f"{selected_model}.joblib")
        st.success(f"Model saved as {selected_model}.joblib")

