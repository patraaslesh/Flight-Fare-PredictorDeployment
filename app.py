import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>✈ Flight Fare Prediction</h1>
<p style='text-align: center; font-size:18px;'>Predict your flight ticket price instantly</p>
""", unsafe_allow_html=True)

st.divider()


# Title
st.title("✈ Flight Fare Prediction")
st.write("Enter flight details to predict ticket price")

# Load saved files
try:
    model = pickle.load(open("flight_price_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# Dropdowns with actual names (NOT numbers)
airline = st.selectbox(
    "Select Airline",
    list(label_encoders['Airline'].classes_)
)

source = st.selectbox(
    "Select Source",
    list(label_encoders['Source'].classes_)
)

destination = st.selectbox(
    "Select Destination",
    list(label_encoders['Destination'].classes_)
)

stops = st.selectbox(
    "Total Stops",
    list(label_encoders['Total_Stops'].classes_)
)

info = st.selectbox(
    "Additional Info",
    list(label_encoders['Additional_Info'].classes_)
)

# Numeric inputs
journey_day = st.number_input("Journey Day", min_value=1, max_value=31, step=1)
journey_month = st.number_input("Journey Month", min_value=1, max_value=12, step=1)

dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23, step=1)
dep_min = st.number_input("Departure Minute", min_value=0, max_value=59, step=1)

arrival_hour = st.number_input("Arrival Hour", min_value=0, max_value=23, step=1)
arrival_min = st.number_input("Arrival Minute", min_value=0, max_value=59, step=1)

duration = st.number_input("Duration (minutes)", min_value=0, step=1)


# Prediction button
if st.button("Predict Price 💰"):

    try:
        # Encode categorical inputs
        airline_encoded = label_encoders['Airline'].transform([airline])[0]
        source_encoded = label_encoders['Source'].transform([source])[0]
        dest_encoded = label_encoders['Destination'].transform([destination])[0]
        stops_encoded = label_encoders['Total_Stops'].transform([stops])[0]
        info_encoded = label_encoders['Additional_Info'].transform([info])[0]

        # Create dataframe in correct column order
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

        # Predict
        prediction = model.predict(input_data)[0]

        # Show result
        st.success(f"💰 Predicted Flight Price: ₹ {int(prediction)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
