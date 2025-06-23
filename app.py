import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    with open('svc_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Create the Streamlit app
st.title('Lung Cancer Prediction System')
st.write("""
This app predicts the likelihood of lung cancer based on various health and lifestyle factors.
""")

# Add an image (optional)
# image = Image.open('lung_cancer_image.jpg')
# st.image(image, caption='Lung Health Awareness', use_column_width=True)

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
        'FATIGUE ': 1 if fatigue == 'Yes' else 0,
        'ALLERGY ': 1 if allergy == 'Yes' else 0,
        'WHEEZING': 1 if wheezing == 'Yes' else 0,
        'ALCOHOL CONSUMING': 1 if alcohol == 'Yes' else 0,
        'COUGHING': 1 if coughing == 'Yes' else 0,
        'SHORTNESS OF BREATH': 1 if shortness == 'Yes' else 0,
        'SWALLOWING DIFFICULTY': 1 if swallowing == 'Yes' else 0,
        'CHEST PAIN': 1 if chest_pain == 'Yes' else 0
    }
    
    # Convert to DataFrame with correct column order
    features = pd.DataFrame([input_data])
    
    # Ensure column order matches training data
    feature_order = [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 
        'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]
    
    return features[feature_order]

# Prediction button
if st.button('Predict Lung Cancer Risk'):
    input_df = prepare_input()
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    st.subheader('Prediction Results')
    
    if prediction[0] == 1:
        st.error('High risk of lung cancer detected')
    else:
        st.success('Low risk of lung cancer detected')
    
    st.write(f"Probability of No Lung Cancer: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Lung Cancer: {prediction_proba[0][1]:.2f}")
    
    # Interpretation
    st.subheader('Interpretation')
    if prediction[0] == 1:
        st.warning("""
        The model predicts a high risk of lung cancer. 
        Please consult with a healthcare professional for further evaluation.
        """)
    else:
        st.info("""
        The model predicts a low risk of lung cancer.
        However, regular check-ups are still recommended for maintaining good health.
        """)

# Add some information about risk factors
st.sidebar.header('About')
st.sidebar.info("""
This prediction model is based on machine learning and is not a substitute for professional medical advice. 
Always consult with a healthcare provider for medical concerns.
""")

st.sidebar.header('Risk Factors')
st.sidebar.write("""
Common risk factors for lung cancer include:
- Smoking
- Exposure to secondhand smoke
- Exposure to radon gas
- Exposure to asbestos and other carcinogens
- Family history of lung cancer
""")
