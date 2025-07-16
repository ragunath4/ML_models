from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


# Sample data (e.g., house size vs. house price)
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print RMSE
print("Root Mean Square Error (RMSE):", round(rmse, 2))

#r2 square value
r2 = r2_score(y_test, y_pred)
print("R² Score (Accuracy):", round(r2 * 100, 2), "%")


# Display test inputs and outputs
results = pd.DataFrame({
    "X_test (sq.ft)": X_test.flatten(),
    "y_test (Actual Price)": y_test,
    "y_pred (Predicted Price)": y_pred.round(2)
})

print("\nModel Test Results:\n")
print(results.to_string(index=False))
