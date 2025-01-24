import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained GBR model
model = joblib.load('trained_Delivery_Time.pkl')

# Apply custom styles
st.markdown(
    """
    <style>
    .main-header {
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 20px;
        color: #333333;
        text-align: center;
        margin-top: -10px;
    }
    .sidebar-section {
        font-size: 16px;
        color: #444444;
        margin-bottom: 20px;
    }
    .prediction-result {
        font-size: 24px;
        color: #1E90FF;
        text-align: center;
        font-weight: bold;
    }
    .success-box {
        background-color: #DFF2BF;
        color: #4F8A10;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">Delivery Time Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting delivery times</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section">Input the details below:</div>', unsafe_allow_html=True)

def Deliver_variables():
    distance_km = st.sidebar.slider('Distance (km)', 0.0, 50.0, 5.0, step=0.1)
    preparation_time_min = st.sidebar.slider('Preparation Time (minutes)', 0.0, 120.0, 15.0, step=1.0)
    courier_experience_yrs = st.sidebar.slider('Courier Experience (years)', 0.0, 30.0, 2.0, step=0.1)

    vehicle_type = st.sidebar.selectbox('Vehicle Type', ["Bike", "Car", "Scooter"])
    vehicle_type_features = {
        "Bike": [0, 1, 0],
        "Car": [0, 0, 1],
        "Scooter": [1, 0, 0],
    }
    vehicle_type_values = vehicle_type_features[vehicle_type]

    weather = st.sidebar.selectbox('Weather', ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
    weather_features = {
        "Clear": [1, 0, 0, 0, 0],
        "Foggy": [0, 1, 0, 0, 0],
        "Rainy": [0, 0, 1, 0, 0],
        "Snowy": [0, 0, 0, 1, 0],
        "Windy": [0, 0, 0, 0, 1],
    }
    weather_values = weather_features[weather]

    traffic_level = st.sidebar.selectbox('Traffic Level', ["Low", "Medium", "High"])
    traffic_level_features = {
        "Low": [0, 1, 0],
        "Medium": [0, 0, 1],
        "High": [1, 0, 0],
    }
    traffic_level_values = traffic_level_features[traffic_level]

    data = {
        'Distance (km)': distance_km,
        'Preparation Time (min)': preparation_time_min,
        'Courier Experience (yrs)': courier_experience_yrs,
        'Vehicle Type': vehicle_type,
        'Weather': weather,
        'Traffic Level': traffic_level
    }

    input_features = (
        [distance_km, preparation_time_min, courier_experience_yrs]
        + vehicle_type_values
        + weather_values
        + traffic_level_values
    )
    return data, input_features

data, input_features = Deliver_variables()

st.subheader('User Input Parameters')
user_input_df = pd.DataFrame([data])
st.table(user_input_df.style.set_properties(**{'text-align': 'center'}))

if st.button("Predict Delivery Time"):
    prediction = model.predict([input_features])[0]
    st.markdown('<div class="prediction-result">Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="success-box">Predicted Delivery Time: {prediction:.2f} minutes</div>', unsafe_allow_html=True)
