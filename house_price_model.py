import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

# Features (X) and Target (y)
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Evaluation")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Example Prediction
sample_house = [[2500, 4, 3]]
predicted_price = model.predict(sample_house)
print("Predicted Price for 2500 sqft house:", predicted_price[0])
