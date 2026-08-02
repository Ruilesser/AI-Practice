import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.random.rand(100, 1) * 100 # Generate 100 random values between 0 and 100
y = 3 * X + np.random.randn(100, 1) * 2 # Linear relationship with some noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test) # predict based off of testing set

# Print coefficients
print("Slope: ", model.coef_[0][0])
print("Intercept: ", model.intercept_[0])

plt.scatter(X_test, y_test, color = 'blue', label = 'Actual')
plt.plot(X_test, y_pred, color = "Red", label = 'Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Performance evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("R-Squared: ", r2)