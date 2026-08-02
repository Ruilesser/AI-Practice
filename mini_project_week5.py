# Do preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Define features and target variable
X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df[['MedHouseVal']]

# Print data information
#print(df.info())
#print(df.describe())

# Visualize the distribution of the target variable
#sns.pairplot(df, vars=['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal'])
#plt.show()

# Check for missing values
#print(df.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")