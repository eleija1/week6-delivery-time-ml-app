import streamlit as st
import pandas as pd
import pickle

st.title("📦 Delivery Time Predictor")

# Load model
model_data = pickle.load(open("simple_week6_voting_model.pkl", "rb"))
model = model_data["model"]
features = model_data["features"]

st.write("Enter order details:")

purchase_dow = st.slider("Day of Week (0=Mon)", 0, 6, 1)
purchase_month = st.slider("Month", 1, 12, 1)
year = st.number_input("Year", value=2026)
product_size_cm3 = st.number_input("Product Size (cm³)", value=5000.0)
product_weight_g = st.number_input("Weight (g)", value=1000.0)
distance_km = st.number_input("Distance (km)", value=10.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[
        purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        distance_km
    ]], columns=features)

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated delivery time: {round(prediction,2)} days")
