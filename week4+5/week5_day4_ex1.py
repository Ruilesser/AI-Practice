import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Get dataset
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2) * 10  # Generate 200 random values between 0 and 10 for two features
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int)  # Create a binary target variable based on a linear combination of features

# Create a Dataframe
df = pd.DataFrame(X, columns=['Age', 'Salary'])
df['Purchase'] = y

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchase'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict probatilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolors='k', cmap="coolwarm")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()