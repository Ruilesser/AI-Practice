import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display information
print("Dataset Info:")
print(df.info())

# Display first few rows
print("\nDataset Head:")
print(df.head())

# Separate features
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# Display summary statistics for features
print("\nSummary Statistics for Categorical Features:")
for feature in categorical_features:
    print(f"\n{feature}:")
    print(df[feature].value_counts())

print("\nSummary Statistics for Numerical Features:")
print(df[numerical_features].describe())