import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained GBR model
model = joblib.load('trained_Delivery_Time.pkl')

# Streamlit app title
st.write("""
# Delivery Time Prediction App
This app predicts the **Delivery Time** for food orders using a Gradient Boosting Regressor (GBR).
""")

st.sidebar.header('Delivery Variables')

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
st.write(pd.DataFrame([data]))

if st.button("Predict Delivery Time"):
    prediction = model.predict([input_features])[0]
    st.subheader('Prediction')
    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")
