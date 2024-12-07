import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
from cv_chooser import cv_chooser
import regressors as rgr
import classifiers as clf

if 'DF' not in st.session_state:
    st.session_state.DF = None
if 'model_type' not in st.session_state:  # ["regressor", "classifier"]
    st.session_state.model_type = None
if 'cv_type' not in st.session_state:
    st.session_state.cv_type = None
if 'cv_args' not in st.session_state:
    st.session_state.cv_args = {}
if 'cv' not in st.session_state:
    st.session_state.cv = None
if 'classy_models' not in st.session_state:
    st.session_state.classy_models = {}
if 'reggy_models' not in st.session_state:
    st.session_state.reggy_models = {}
if 'classy_scores' not in st.session_state:
    st.session_state.classy_scores = {}
if 'reggy_scores' not in st.session_state:
    st.session_state.reggy_scores = {}
if 'x' not in st.session_state:
    st.session_state.x = None
if 'y' not in st.session_state:
    st.session_state.y = None

def main():
    step = stx.stepper_bar(steps=["Upload Data", "Select Sampling Method", "Train Models"])
    match step:
        case 0:
            uploaded_file = st.file_uploader("Upload Data", type=['csv'])
            if uploaded_file is not None:
                st.session_state.DF = pd.read_csv(uploaded_file)
            if st.session_state.DF is not None:
                columns = list(st.session_state.DF.columns)
                st.session_state.y = st.session_state.DF.iloc[:,-1]
                st.session_state.x = st.session_state.DF.iloc[:,:-1]
                
                
                st.write(st.session_state.DF)
        case 1:
            if st.session_state.DF is not None:
                cv_chooser()
            else:
                st.warning("Please upload data first")
        case 2:
            if st.session_state.DF is not None and st.session_state.cv is not None:
                with st.sidebar:
                    model_type = st.radio("Select model type:", ["Regression", "Classification"])
                    models = []
                    pkg = None
                    if model_type == "Regression":
                        pkg = rgr
                        models = rgr.MODEL_MAPPING.keys()
                    else:
                        pkg = clf
                        models = clf.MODEL_MAPPING.keys()
                    for model_name in models:
                        expander = st.expander(model_name)
                        with expander:
                            pkg.MODEL_PARAMS[model_name]()

                pkg.show_graph()

            else:
                st.warning("Please upload data and select sampling method first")

if __name__ == "__main__":
    main()
