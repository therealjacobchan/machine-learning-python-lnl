import numpy as np
from sklearn.linear_model import LinearRegression

# Input features
X = np.array([
    [1500, 3, 2, 0.5],
    [2000, 4, 3, 1.0],
    [1200, 2, 1, 0.3],
    [1800, 3, 2, 0.7],
    [2100, 4, 3, 1.2],
    [1600, 3, 2, 0.6],
    [1900, 3, 2, 0.8]
])

# Output labels
y = np.array([300, 400, 250, 350, 425, 320, 380])

# Create and fit the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Example of prediction for a new house
new_house = np.array([[1700, 3, 2, 0.4]])  # New house features
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price)
