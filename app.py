import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("model.joblib")
feature_names = joblib.load("features_name.joblib")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Enter property details to estimate the house price.")

st.sidebar.header("Enter House Details")

# Input fields based on your dataset
area = st.sidebar.number_input(
    "Area (square feet)", min_value=100, value=5000
)
bedrooms = st.sidebar.number_input(
    "Bedrooms", min_value=1, max_value=20, value=3
)
bathrooms = st.sidebar.number_input(
    "Bathrooms", min_value=1, max_value=20, value=2
)
stories = st.sidebar.number_input(
    "Stories", min_value=1, max_value=10, value=2
)

mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])
hotwater = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
parking = st.sidebar.number_input(
    "Parking Spaces", min_value=0, max_value=10, value=1
)
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# Prepare inputs using the dataset's column names
input_data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwater,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}

input_df = pd.DataFrame([input_data])

# Match the training dataset's encoded columns
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(
    columns=feature_names, fill_value=0
)

st.subheader("Selected Property Details")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Price", type="primary"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated House Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")