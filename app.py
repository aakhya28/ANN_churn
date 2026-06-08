import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import pickle

model=tf.keras.models.load_model('model.h5')
#load encoder and scaler
with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file)

with open('le_gender.pkl','rb') as file:
    le_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer Churn Prediction')

geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10,5)
num_of_products= st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_memeber=st.selectbox('Is Active member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le_gender.transform([[gender]])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_memeber],
    'EstimatedSalary':[estimated_salary]
})

geo_ohe=ohe.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_ohe,columns=ohe.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_scaled=scaler.transform(input_data)

prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]
st.write(f"Churn probability:{prediction_proba:.2f}")

if prediction_proba>0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is not likely to churn")
