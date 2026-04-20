import pandas as pd 
import numpy as np
import streamlit as st
import tensorflow as tf
import pickle
import joblib

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model

@st.cache_data
def Load_Data(Csv):
    return pd.read_csv(Csv)

Model = joblib.load('Model_Pipeline.pkl')

Train_Metrics = joblib.load('Train_Metrics.pkl')

Test_Metrics = joblib.load('Test_Metrics.pkl')

@st.cache_resource
def Trained_Model(data):
    return Model, Train_Metrics, Test_Metrics

st.sidebar.header('(1) Load Data')

Csv = st.sidebar.text_input(
    'Csv Path',
    value='Sales_.csv',
    help='Put The csv file name into the folder'
)

try:
    Data = Load_Data(Csv)
    st.sidebar.success(f'Data Loaded {len(Data)} Rows')

except Exception as e:
    st.sidebar.error('Data Not Loaded')
    st.stop()

st.sidebar.divider()

st.sidebar.header('(2) Train Model')

if st.sidebar.button('Train Re / Train'):
    st.cache_resource.clear()
    st.sidebar.success('Cache Reset')

Model, Test_Metrics, Train_Metrics = Trained_Model(Data)

st.set_page_config(page_title='Sales Forecasting Prediction', layout='wide')
st.title('Sales Prediction & Analysis Dashboard')
st.caption('an End To End Machine Learning Data Science With Ai Project To Analyzed actual Sales')


st.subheader('Data Preview')

st.write(Data.head())

cola,colb = st.columns(2)

with cola:

    st.subheader('Train-Metrics')
    
    Train = pd.DataFrame(
        [Train_Metrics['mae'],Train_Metrics['mse'],round(Train_Metrics['r2'],2)],
        index=['mean absolute error','mean squared error','r2 score'],
        columns=['Metrics']
    )

    st.write(Train)

with colb:

    st.subheader('Test-Metrics')

    Test = pd.DataFrame(
        [Test_Metrics['mae'],Test_Metrics['mse'],round(Test_Metrics['r2'],2)],
        index=['mean absolute error','mean squared error','r2 score'],
        columns=['Metrics'],
    )

    st.write(Test)

st.subheader('User Input')

col1,col2,col3 = st.columns(3)

with col1:
    
    Segment = st.selectbox('Segment',[1,2,3])
    Country = st.selectbox('Country', sorted(Data['Country'].unique()))
    City = st.selectbox('City', sorted(Data['City'].unique()))
    State = st.selectbox('State', sorted(Data['State'].unique()))
    Region = st.selectbox('Region', sorted(Data['Region'].unique()))

with col2:

    Category = st.selectbox('Category', sorted(Data['Category'].unique()))
    Sub_Category = st.selectbox('Sub_Category', sorted(Data['Sub_Category'].unique()))
    Quantity = st.slider('Quantity',50,200,100)
    Cost = st.number_input('Product-Cost', max_value = 300, min_value=100, value = 200, step = 20)
    
with col3:

    Order_Day = st.slider('Order_Day',1,31,6)
    Order_Month = st.selectbox('Order_Month', ['January','February','March','April','May','June','July','August','September','October','November','December'])
    Order_Year = st.selectbox('Order_Year',[2021,2022,2023,2024])
    Week_No = st.selectbox('Week_No',[1,2,3,4,5,6])    
        

User_Input = pd.DataFrame({

    'Segment' : [Segment],
    'Country' : [Country],
    'City' : [City],
    'State' : [State],
    'Region' : [Region],
    'Category' : [Category],
    'Sub_Category' : [Sub_Category],
    'Quantity' : [Quantity],
    'Cost' : [Cost],
    'Order_Day' : [Order_Day],
    'Order_Month' : [Order_Month],
    'Order_Year' : [Order_Year],
    'Week_No' : [Week_No],

})

st.divider()

if st.button('Predict Sales'):
    Prediction = Model.predict(User_Input)

    Prediction = Prediction[0]

    st.success(f"Expected Sales is PKR : {Prediction:,.2f}")

    st.subheader('User Input detail')

    st.write(User_Input)
