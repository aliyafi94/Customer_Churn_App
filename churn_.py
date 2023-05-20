import os
import streamlit as st
import pandas as pd
import pickle


# Model and System
MAIN_PATH  = os.path.abspath(os.getcwd())
PATH_MODEL = os.path.join(MAIN_PATH, "final_model.sav")
print(MAIN_PATH)
lgbm = pickle.load(open(PATH_MODEL, 'rb'))
feature = pd.DataFrame({
    'NumberOfDeviceRegistered':[],
    'SatisfactionScore':[],
    'NumberOfAddress':[],
    'Complain':[],
    'Tenure':[],
    'WarehouseToHome':[],
    'DaySinceLastOrder':[],
    'CashbackAmount':[],
    'PreferedOrderCat':[],
    'MaritalStatus':[],
})


# Title
st.set_page_config(page_title="Customer Churn Prediction", initial_sidebar_state="collapsed")
st.header('Customer Churn Prediction')
st.write("""
         The Customer Churn Prediction App is a web-based application that helps employers predict the Churning of their Customers. 
         The app uses machine learning algorithms to analyze a Customers's transaction history 
         and other relevant data points to determine their likelihood of staying with the E-Commerce Service for an extended period.
         """)
st.title('Customer Data')

# House feature
col1, col2 = st.columns(2)

with col1:
    name = st.text_input('Name')
    tenure = st.number_input(label="Tenure (Month)", min_value=0, step=1)
    wh = st.number_input(label="Warehouse to home (km)", min_value=1, step=1)
    status = st.selectbox("Marital Status", ('Single', 'Married', 'Divorced'))
    address = st.number_input(label="Number of Address", min_value=1, step=1)
    device = st.number_input(label="Number of Device Registered", min_value=1, step=1)
with col2:
    id = st.number_input('Customer ID', min_value=1)
    score = st.selectbox("Satisfaction Score", (1, 2, 3, 4, 5))
    last_order = st.number_input(label="Day Since Last Order", min_value=0,step=1)
    cashback = st.number_input(label="Cashback Amount ($)", min_value=0.00, step=0.01)
    complain = st.selectbox("Complain (0 = No Complain, 1 = Complain)", (0, 1))
    category = st.selectbox("Prefered Order Category", ('Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Others', 'Grocery'))

pred_process = st.button("Predict",use_container_width=True)

if pred_process:
    feature.loc[0, 'NumberOfDeviceRegistered'] = device
    feature.loc[0,'SatisfactionScore'] = score
    feature.loc[0,'NumberOfAddress'] = address
    feature.loc[0,'Complain'] = complain
    feature.loc[0,'Tenure'] = tenure
    feature.loc[0,'WarehouseToHome'] = wh
    feature.loc[0,'DaySinceLastOrder'] = last_order
    feature.loc[0,'CashbackAmount'] = cashback
    feature.loc[0,'PreferedOrderCat'] = category
    feature.loc[0,'MaritalStatus'] = status
    
    print(feature)
    prob = lgbm.predict_proba(feature)

    photo, information = st.columns(2)
       
    with information:
        st.write("Name          : ", name)
        st.write("Customer ID   : ", id)
        st.write("Churn Probability   : ", str(round(prob[:,1][0], 2)))