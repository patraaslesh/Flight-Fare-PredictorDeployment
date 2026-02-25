import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>✈ Flight Fare Prediction</h1>", unsafe_allow_html=True)
st.write("Enter flight details to predict ticket price")

# Load saved files
try:
    model = pickle.load(open("flight_price_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Dropdowns
airline = st.selectbox("Select Airline", list(label_encoders['Airline'].classes_))
source = st.selectbox("Select Source", list(label_encoders['Source'].classes_))
destination = st.selectbox("Select Destination", list(label_encoders['Destination'].classes_))
stops = st.selectbox("Total Stops", list(label_encoders['Total_Stops'].classes_))
info = st.selectbox("Additional Info", list(label_encoders['Additional_Info'].classes_))

# Numeric inputs
journey_day = st.number_input("Journey Day", 1, 31)
journey_month = st.number_input("Journey Month", 1, 12)
dep_hour = st.number_input("Departure Hour", 0, 23)
dep_min = st.number_input("Departure Minute", 0, 59)
arrival_hour = st.number_input("Arrival Hour", 0, 23)
arrival_min = st.number_input("Arrival Minute", 0, 59)
duration = st.number_input("Duration (minutes)", 0)

# Predict
if st.button("Predict Price 💰"):

    try:
        airline_encoded = label_encoders['Airline'].transform([airline])[0]
        source_encoded = label_encoders['Source'].transform([source])[0]
        dest_encoded = label_encoders['Destination'].transform([destination])[0]
        stops_encoded = label_encoders['Total_Stops'].transform([stops])[0]
        info_encoded = label_encoders['Additional_Info'].transform([info])[0]

        input_data = pd.DataFrame([[
            airline_encoded,
            source_encoded,
            dest_encoded,
            stops_encoded,
            info_encoded,
            journey_day,
            journey_month,
            dep_hour,
            dep_min,
            arrival_hour,
            arrival_min,
            duration
        ]], columns=feature_columns)

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Flight Price: ₹ {int(prediction)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
