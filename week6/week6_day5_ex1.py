import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('week6/bike-sharing-daily.csv')

# Display information
print("Dataset Info:")
print(df.info())

# Preview the first few rows of the dataset
print("\nDataset Head:")
print(df.head())

# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# Display the updated dataset information
print("\nUpdated Dataset Info:")
print(df[['dteday', 'day_of_week', 'month', 'year']].head())

# Select features and target variable
X = df[['temp']]
y = df['cnt'] # count of total rental bikes including both casual and registered

# Apply polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Display the transformed features
print("\nOriginal and Transformed Features (Polynomial):")
print(pd.DataFrame(X_poly, columns=['temp', 'temp_squared']).head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state=42)

# Train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)

# Train and evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Compare the mean squared errors
print(f"\nMean Squared Error with Original Features: {mse_original:.2f}")
print(f"Mean Squared Error with Polynomial Features: {mse_poly:.2f}")