import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


data = pd.read_csv("MagicBricks.csv")
data.dropna(inplace=True)
data = data.drop('Locality', axis=1)


numerical_cols = list(data.select_dtypes(include=np.number))
categorical_cols = list(data.select_dtypes(include=object))
data_cat_encoded = pd.get_dummies(data[categorical_cols])

final_data = pd.concat([data[numerical_cols], data_cat_encoded], axis=1)
y = final_data['Price']
x = final_data.drop('Price', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)


st.title("House Price Prediction")
st.sidebar.header("Input Features")


area = st.sidebar.number_input("Area (in square feet)", min_value=0.0, step=0.1)
bhk = st.sidebar.slider("Number of Bedrooms (BHK)", 1, 10)
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 10)
parking = st.sidebar.slider("Parking Spaces", 0, 5)
per_sqft = st.sidebar.number_input("Price per Square Foot", min_value=0.0, step=0.1)

furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
status = st.sidebar.selectbox("Status", ["Almost Ready", "Ready to Move"])
transaction = st.sidebar.selectbox("Transaction", ["New Property", "Resale"])
property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Builder Floor"])


furnishing_furnished = 1 if furnishing == "Furnished" else 0
furnishing_semi_furnished = 1 if furnishing == "Semi-Furnished" else 0
furnishing_unfurnished = 1 if furnishing == "Unfurnished" else 0

status_almost_ready = 1 if status == "Almost Ready" else 0
status_ready_to_move = 1 if status == "Ready to Move" else 0

transaction_new_property = 1 if transaction == "New Property" else 0
transaction_resale = 1 if transaction == "Resale" else 0

property_type_apartment = 1 if property_type == "Apartment" else 0
property_type_builder_floor = 1 if property_type == "Builder Floor" else 0


input_data = pd.DataFrame({
    "Area": [area],
    "BHK": [bhk],
    "Bathroom": [bathroom],
    "Parking": [parking],
    "Per_Sqft": [per_sqft],
    "Furnishing_Furnished": [furnishing_furnished],
    "Furnishing_Semi-Furnished": [furnishing_semi_furnished],
    "Furnishing_Unfurnished": [furnishing_unfurnished],
    "Status_Almost_ready": [status_almost_ready],
    "Status_Ready_to_move": [status_ready_to_move],
    "Transaction_New_Property": [transaction_new_property],
    "Transaction_Resale": [transaction_resale],
    "Type_Apartment": [property_type_apartment],
    "Type_Builder_Floor": [property_type_builder_floor]
})


predicted_price = gbr.predict(input_data)


st.subheader(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
