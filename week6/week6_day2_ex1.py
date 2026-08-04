from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display information
print("Dataset Info:")
print(X.describe())
print("\nTarget Classes:", data.target_names)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy without scaling:", accuracy)

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled dataset into training and testing sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)

# Make predictions on scaled data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Calculate accuracy for scaled data
accuracy_scaled = accuracy_score(y_test_scaled, y_pred_scaled)
print("Accuracy with scaling:", accuracy_scaled)

# Apply Standard Scaling
standard_scaler = StandardScaler()
X_standard_scaled = standard_scaler.fit_transform(X)

# Split the standard scaled dataset into training and testing sets
X_train_standard, X_test_standard, y_train_standard, y_test_standard = train_test_split(X_standard_scaled, y, test_size=0.2, random_state=42)

# Train the classifier on standard scaled data
knn_standard_scaled = KNeighborsClassifier(n_neighbors=5)
knn_standard_scaled.fit(X_train_standard, y_train_standard)

# Make predictions on standard scaled data
y_pred_standard_scaled = knn_standard_scaled.predict(X_test_standard)

# Calculate accuracy for standard scaled data
accuracy_standard_scaled = accuracy_score(y_test_standard, y_pred_standard_scaled)
print("Accuracy with standard scaling:", accuracy_standard_scaled)