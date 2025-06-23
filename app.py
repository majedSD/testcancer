import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model with better error handling
@st.cache_resource
def load_model():
    try:
        with open('svc_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Verify the loaded model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid scikit-learn model")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

# Create the Streamlit app
st.title('Lung Cancer Prediction System')
st.write("""
This app predicts the likelihood of lung cancer based on various health and lifestyle factors.
""")

# Input fields
st.header('Patient Information')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.radio('Gender', ['Male', 'Female'])
    age = st.slider('Age', 20, 100, 50)
    smoking = st.selectbox('Smoking', ['No', 'Yes'])
    yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'])
    anxiety = st.selectbox('Anxiety', ['No', 'Yes'])
    peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'])
    chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'])

with col2:
    fatigue = st.selectbox('Fatigue', ['No', 'Yes'])
    allergy = st.selectbox('Allergy', ['No', 'Yes'])
    wheezing = st.selectbox('Wheezing', ['No', 'Yes'])
    alcohol = st.selectbox('Alcohol Consuming', ['No', 'Yes'])
    coughing = st.selectbox('Coughing', ['No', 'Yes'])
    shortness = st.selectbox('Shortness of Breath', ['No', 'Yes'])
    swallowing = st.selectbox('Swallowing Difficulty', ['No', 'Yes'])
    chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'])

# Convert inputs to model format
def prepare_input():
    # Convert to binary (1 for Yes, 0 for No)
    input_data = {
        'GENDER': 1 if gender == 'Male' else 0,
        'AGE': age,
        'SMOKING': 1 if smoking == 'Yes' else 0,
        'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
        'ANXIETY': 1 if anxiety == 'Yes' else 0,
        'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
        'CHRONIC DISEASE': 1 if chronic_disease == 'Yes' else 0,
        'FATIGUE': 1 if fatigue == 'Yes' else 0,
        'ALLERGY': 1 if allergy == 'Yes' else 0,
        'WHEEZING': 1 if wheezing == 'Yes' else 0,
        'ALCOHOL CONSUMING': 1 if alcohol == 'Yes' else 0,
        'COUGHING': 1 if coughing == 'Yes' else 0,
        'SHORTNESS OF BREATH': 1 if shortness == 'Yes' else 0,
        'SWALLOWING DIFFICULTY': 1 if swallowing == 'Yes' else 0,
        'CHEST PAIN': 1 if chest_pain == 'Yes' else 0
    }
    
    # Convert to numpy array in correct order
    feature_order = [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
        'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]
    
    return np.array([[input_data[col] for col in feature_order]])

# Prediction button
if st.button('Predict Lung Cancer Risk'):
    try:
        input_data = prepare_input()
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction = np.array(prediction).flatten()
        
        st.subheader('Prediction Results')
        
        if prediction.size == 0:
            st.error("No prediction was returned")
        elif prediction[0] == 1:
            st.error('High risk of lung cancer detected')
            risk_level = "High"
        else:
            st.success('Low risk of lung cancer detected')
            risk_level = "Low"
        
        # Get confidence scores if available
        try:
            if hasattr(model, 'decision_function'):
                confidence = model.decision_function(input_data)
                st.write(f"Confidence score: {confidence[0]:.2f}")
                st.write("""
                Interpretation:
                - Positive values indicate higher risk
                - Negative values indicate lower risk
                - Magnitude indicates confidence
                """)
            else:
                st.info("Probability estimates not available for this model configuration")
        except Exception as e:
            st.warning(f"Couldn't get confidence scores: {str(e)}")
        
        # Recommendations
        st.subheader('Recommendation')
        if risk_level == "High":
            st.warning("""
            **Consult a healthcare professional
