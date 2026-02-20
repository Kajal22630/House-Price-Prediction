import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("house_data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

st.title("🏠 House Price Prediction App")

area = st.number_input("Enter Area (sq ft)", min_value=500)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1)
bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1)

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Predicted House Price: ₹ {int(prediction[0])}")
