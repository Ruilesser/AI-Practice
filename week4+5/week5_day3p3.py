import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame

# Select features (Median Income) and Target (Median House Value)
X = df[['MedInc']]
y = df[['MedHouseVal']]

# Transform features to polynomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

"""
# Firt polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make Predictions
y_pred = model.predict(X_poly)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label = "Actual Data", alpha=0.5)
plt.scatter(X, y_pred, color="red", label = "Predicted Data", alpha=0.5)
plt.title("Polynomial Regression")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()

# Evaluate model performance
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error: ", mse)
"""

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Evaluate Regression
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE: ", ridge_mse)

lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE: ", lasso_mse)

# Visualize Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color="blue", label="Actual Data", alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color="green", label="Ridge Predictions", alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color="orange", label="Lasso Predictions", alpha=0.5)
plt.title("Ridge vs Lasso Regression Comparison")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()