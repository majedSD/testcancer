import streamlit as st
import numpy as np
import pickle

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    try:
        with open('Random_Forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid model loaded")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

st.title('Lung Cancer Prediction System')
st.write("This app predicts lung cancer risk based on health and lifestyle indicators.")

# User input
st.header('Patient Information')
col1, col2 = st.columns(2)

with col1:
    gender = st.radio('Gender', ['Male', 'Female'], index=0)
    age = st.slider('Age', 20, 100, 60)
    smoking = st.selectbox('Smoking', ['No', 'Yes'], index=1)
    yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'], index=1)
    anxiety = st.selectbox('Anxiety', ['No', 'Yes'], index=0)
    peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'], index=0)
    chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'], index=1)

with col2:
    fatigue = st.selectbox('Fatigue', ['No', 'Yes'], index=1)
    allergy = st.selectbox('Allergy', ['No', 'Yes'], index=0)
    wheezing = st.selectbox('Wheezing', ['No', 'Yes'], index=1)
    alcohol = st.selectbox('Alcohol Consuming', ['No', 'Yes'], index=0)
    coughing = st.selectbox('Coughing', ['No', 'Yes'], index=1)
    shortness = st.selectbox('Shortness of Breath', ['No', 'Yes'], index=1)
    swallowing = st.selectbox('Swallowing Difficulty', ['No', 'Yes'], index=0)
    chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'], index=1)

# Correct mapping to training values: No=1, Yes=2
def encode_yes_no(val):
    return 2 if val == 'Yes' else 1

def prepare_input():
    input_data = [
        1 if gender == 'Male' else 0,
        age,
        encode_yes_no(smoking),
        encode_yes_no(yellow_fingers),
        encode_yes_no(anxiety),
        encode_yes_no(peer_pressure),
        encode_yes_no(chronic_disease),
        encode_yes_no(fatigue),
        encode_yes_no(allergy),
        encode_yes_no(wheezing),
        encode_yes_no(alcohol),
        encode_yes_no(coughing),
        encode_yes_no(shortness),
        encode_yes_no(swallowing),
        encode_yes_no(chest_pain)
    ]
    return np.array([input_data])

# Predict
if st.button('Predict Lung Cancer Risk'):
    try:
        input_data = prepare_input()
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("High risk of lung cancer detected")
            risk_level = "High"
        else:
            st.success("Low risk of lung cancer detected")
            risk_level = "Low"

        # Confidence (if available)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)
            st.write(f"Risk Probability: {proba[0][1]:.1%}")

        # Recommendations
        st.subheader('Recommendation')
        if risk_level == "High":
            st.warning("""
            **Consult a healthcare professional immediately**  
            • Schedule a doctor's appointment  
            • Consider diagnostic tests  
            • Review risk factors
            """)
        else:
            st.info("""
            **Maintain healthy habits**  
            • Regular check-ups recommended  
            • Avoid smoking  
            • Monitor for symptoms
            """)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.sidebar.header('About')
st.sidebar.info("This tool provides estimates only. Always consult a healthcare professional for medical advice.")
