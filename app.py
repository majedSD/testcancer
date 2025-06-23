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
    
    # Display results
    st.subheader('Prediction Results')
    
    if prediction[0] == 1:
        st.error('High risk of lung cancer detected')
        risk_level = "High"
    else:
        st.success('Low risk of lung cancer detected')
        risk_level = "Low"
    
    # Try different methods to get confidence information
    try:
        # Method 1: Check for decision_function
        if hasattr(model, 'decision_function'):
            confidence = model.decision_function(input_df)
            st.write(f"Confidence score: {confidence[0]:.2f}")
            if abs(confidence[0]) > 1:
                st.write("Higher absolute values indicate stronger confidence")
        
        # Method 2: Check for predict_proba (if enabled)
        elif hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_df)
            st.write(f"Probability of Lung Cancer: {prediction_proba[0][1]:.2%}")
        
        # Method 3: If neither is available
        else:
            st.info("This model provides binary predictions without confidence scores")
    
    except Exception as e:
        st.warning(f"Could not get confidence metrics: {str(e)}")
    
    # Interpretation
    st.subheader('Recommendation')
    if risk_level == "High":
        st.warning("""
        **Consult a healthcare professional immediately**  
        - Schedule a doctor's appointment  
        - Consider diagnostic tests like CT scans  
        - Review risk factors and lifestyle changes
        """)
    else:
        st.info("""
        **Maintain healthy habits**  
        - Regular health check-ups recommended  
        - Avoid smoking and secondhand smoke  
        - Monitor for any new symptoms
        """)

# Add some information about risk factors
st.sidebar.header('About')
st.sidebar.info("""
This prediction model is based on machine learning and is not a substitute for professional medical advice. 
Always consult with a healthcare provider for medical concerns.
""")

st.sidebar.header('Risk Factors')
st.sidebar.write("""
Common risk factors for lung cancer:
- Smoking (accounts for 80-90% of cases)
- Exposure to radon gas
- Occupational exposures (asbestos, arsenic)
- Family history of lung cancer
- Previous radiation therapy to the chest
""")

st.sidebar.header('Early Detection')
st.sidebar.write("""
Early signs may include:
- Persistent cough
- Chest pain
- Shortness of breath
- Coughing up blood
- Unexplained weight loss
""")
