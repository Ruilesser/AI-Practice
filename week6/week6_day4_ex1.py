from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# Load the Diabetes dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display information
print("Dataset Info:")
print(df.head())
print(df.info())

correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Diabetes Dataset')
plt.show()

# Select features with high correlation with the target variable
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
print("\nFeatures correlated with target variable:")
print(correlated_features)

# Separate features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Calculate mutual information scores
mutual_info = mutual_info_regression(X, y)

# Create a DataFrame for mutual information scores
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

print("\nMutual Information Scores:")
print(mi_df)

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train a Random Forest Regressor to get feature importances
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importances
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances from Random Forest:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importances from Random Forest')
plt.show()