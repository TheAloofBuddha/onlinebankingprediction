import joblib
import pandas as pd
import streamlit as st
import numpy as np

# Load the model and encoders
model = joblib.load("model.joblib")
location_encoder = joblib.load('location_encoder.joblib')
cellphone_encoder = joblib.load('cellphone_encoder.joblib')
education_level_encoder = joblib.load('education_level_encoder.joblib')
gender_encoder = joblib.load('gender_encoder.joblib')
job_type_encoder = joblib.load('job_type_encoder.joblib')
marital_status_encoder = joblib.load('marital_status_encoder.joblib')
relationship_encoder = joblib.load('relationship_encoder.joblib')
target_encoder = joblib.load('target_encoder.joblib')
country_encoder = joblib.load('country_encoder.joblib')

st.title('Supervised Classification Machine Learning Model')
st.header('customer Online Banking Prediction')

column_desc = pd.read_csv(r'C:\Users\PC\Documents\data science GMC\financial_dataset_model\VariableDefinitions (1).csv',header=1)
st.write("The dataset contains information about customers' online banking usage, including their demographics, account details, and transaction history. The goal is to predict whether a customer will use online banking services or not.")
st.dataframe(column_desc)

with st.sidebar:
    country = st.selectbox('Customer Country', [i for i in country_encoder.classes_])
    location = st.selectbox('Urban or Rural Dwelling', [i for i in location_encoder.classes_])
    cellphone = st.selectbox('Customer Has a Cellphone?', [i for i in cellphone_encoder.classes_])
    age = st.number_input('Customer Age', 1, 100)
    education_level = st.selectbox('Education Level', [i for i in education_level_encoder.classes_])
    gender = st.selectbox('Gender', [i for i in gender_encoder.classes_])
    job_type = st.selectbox('Job Type', [i for i in job_type_encoder.classes_])
    marital_status = st.selectbox('Marital Status', [i for i in marital_status_encoder.classes_])
    relationship = st.selectbox('Relationship to Head of Household', [i for i in relationship_encoder.classes_])
    household_size = st.number_input('Household Size', 1, 100)

features = [
    country_encoder.transform([country]),
    location_encoder.transform([location]),
    cellphone_encoder.transform([cellphone]),
    household_size,
    age,
    gender_encoder.transform([gender]),
    relationship_encoder.transform([relationship]),
    marital_status_encoder.transform([marital_status]),
    education_level_encoder.transform([education_level]),
    job_type_encoder.transform([job_type])
    
    ]

# Convert features to a DataFrame
user_data = pd.DataFrame([features], columns=['country', 'location_type', 'cellphone_access', 'household_size',
       'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
       'marital_status', 'education_level', 'job_type'])

if st.sidebar.button('Predict'):
    st.dataframe(user_data)
    prediction = model.predict(user_data)
    if prediction == 0:
        st.success('Customer is not likely to use online banking')
    else:
        st.success('Customer is likely to use online banking')