import streamlit as st
import joblib
import pandas as pd

# Load the model and other required objects
model = joblib.load('random_forest_model.pkl')


# Define the main function for the app
def main():
    st.title("Online Food Prediction App")
    
    # Input fields
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married"])
    occupation = st.selectbox("Occupation", options=["Student", "Professional", "Self-Employed", "Unemployed"])
    monthly_income = st.selectbox("Monthly Income", options=["No Income", "Below Rs.10000", "Rs.10000-25000", "Rs.25000-50000", "Above Rs.50000"])
    educational_qualifications = st.selectbox("Educational Qualifications", options=["Graduate", "Post Graduate", "PhD"])
    family_size = st.number_input("Family size", min_value=1, max_value=10, value=3)
    latitude = st.number_input("Latitude", value=12.9716)
    longitude = st.number_input("Longitude", value=77.5946)
    feedback = st.selectbox("Feedback", options=["Positive", "Negative"])
    
    # Preprocess the inputs
    inputs = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Occupation': [occupation],
        'Monthly Income': [monthly_income],
        'Educational Qualifications': [educational_qualifications],
        'Family size': [family_size],
        'latitude': [latitude],
        'longitude': [longitude],
        'Feedback': [feedback]
    })
    
    for column in ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']:
        inputs[column] = label_encoders[column].transform(inputs[column])
    
    numerical_columns = ['Age', 'Family size', 'latitude', 'longitude']
    inputs[numerical_columns] = scaler.transform(inputs[numerical_columns])
    
    # Prediction
    if st.button("Predict"):
        prediction = model.predict(inputs)
        st.write("Prediction:", "Yes" if prediction[0] == 1 else "No")

if __name__ == '__main__':
    main()
