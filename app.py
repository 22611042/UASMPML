import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the model
model = joblib.load('random_forest_model.pkl')

# Define label encoders and scaler (they should be the same as used in preprocessing)
categorical_columns = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
label_encoders = {column: LabelEncoder() for column in categorical_columns}

# Fit the encoders with the possible values (these values should match your training data)
label_encoders['Gender'].fit(['Male', 'Female'])
label_encoders['Marital Status'].fit(['Single', 'Married'])
label_encoders['Occupation'].fit(['Student', 'Professional', 'Others'])
label_encoders['Monthly Income'].fit(['No Income', 'Below Rs.10000', 'Rs.10001 to Rs.30000', 'Rs.30001 to Rs.50000', 'Above Rs.50000'])
label_encoders['Educational Qualifications'].fit(['Graduate', 'Post Graduate', 'Others'])
label_encoders['Feedback'].fit(['Positive', 'Negative'])

scaler = StandardScaler()
numerical_columns = ['Age', 'Family size', 'latitude', 'longitude']

# Function to preprocess user input
def preprocess_input(data):
    for column in categorical_columns:
        data[column] = label_encoders[column].transform([data[column]])[0]
    data[numerical_columns] = scaler.transform([data[numerical_columns]])
    return data

# Set up the Streamlit app
st.title('Online Food Service Prediction')
st.write('Enter the details below to predict if a person will use the online food service.')

# User inputs
age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
occupation = st.selectbox('Occupation', ['Student', 'Professional', 'Others'])
monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', 'Rs.10001 to Rs.30000', 'Rs.30001 to Rs.50000', 'Above Rs.50000'])
education = st.selectbox('Educational Qualifications', ['Graduate', 'Post Graduate', 'Others'])
family_size = st.number_input('Family Size', min_value=1, max_value=20, value=4)
latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0)
feedback = st.selectbox('Feedback', ['Positive', 'Negative'])

# Preprocess input
input_data = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': education,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Feedback': feedback
}

input_df = pd.DataFrame([input_data])
processed_input = preprocess_input(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(processed_input)
    if prediction == 1:
        st.success('The person is likely to use the online food service.')
    else:
        st.error('The person is unlikely to use the online food service.')
