import streamlit as st
import joblib

def predict_regression(model, feature_names):
    st.subheader("Make a Prediction")
    inputs = []
    c1, c2 = st.columns(2)
    for feature in feature_names[0:len(feature_names)//2]:
        value = c1.number_input(f"Enter value for {feature}:")
        inputs.append(value)
    
    for feature in feature_names[len(feature_names)//2:]:
        value = c2.number_input(f"Enter value for {feature}:")
        inputs.append(value)
    
    if st.button("Predict"):
        prediction = model.predict([inputs])
        percentage = prediction[0] * 100  # Convert to percentage
        st.write(f"Predicted Probability to Flood: {percentage:.2f}%")

def predict_classification(model, feature_names):
    st.subheader("Make a Prediction")
    inputs = []
    c1, c2 = st.columns(2)
    for feature in feature_names[0:len(feature_names)//2]:
        value = c1.number_input(f"Enter value for {feature}:")
        inputs.append(value)
    
    for feature in feature_names[len(feature_names)//2:]:
        value = c2.number_input(f"Enter value for {feature}:")
        inputs.append(value)
    
    if st.button("Predict"):
            prediction = model.predict([inputs])
            # Interpret the prediction
            if prediction[0] == 0:
                prediction_text = "without heart disease"
            else:
                prediction_text = "with heart disease"

            st.write(f"Predicted Class: {prediction[0]} ({prediction_text})")

def main():
    st.markdown("<h1 style='text-align: center;'>Model Prediction</h1>", unsafe_allow_html=True)
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None

    st.session_state.model_type = st.radio("Select model type:", ["Regression", "Classification"])
    feature_names = {
        "Regression": ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
                        'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
                        'Siltation', 'AgriculturalPractices', 'Encroachments',
                        'IneffectiveDisasterPreparedness', 'DrainageSystems',
                        'CoastalVulnerability', 'Landslides', 'Watersheds',
                        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                        'InadequatePlanning', 'PoliticalFactors'],
        "Classification": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    }
    uploaded_model = st.file_uploader("Upload Model", type=['joblib'])
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        features = feature_names[st.session_state.model_type]
        if st.session_state.model_type == "Regression":
            predict_regression(model, features)
        else:
            predict_classification(model, features)

if __name__ == "__main__":
    main()
