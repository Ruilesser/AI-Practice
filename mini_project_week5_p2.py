import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# PART 2
# Load Telco customer churn dataset
df_telco = pd.read_csv('Telco-Customer-Churn.csv')

# Inspect the dataset
#print(df_telco.info())
#print(df_telco.describe())

# Visualize the distribution of the target variable
#sns.countplot(x='Churn', data=df_telco)
#plt.title('Distribution of Churn')
#plt.show()

# Handle missing values
# df_telco.fillna(df_telco.mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df_telco['Churn'] = le.fit_transform(df_telco['Churn'])

# Define features and target variable
X_telco = df_telco.drop(columns=['Churn'])
X_telco = X_telco.select_dtypes(exclude=['object'])  # keep only numeric features
y_telco = df_telco['Churn']

# Scale the features
scaler = StandardScaler()
X_telco_scaled = scaler.fit_transform(X_telco)

# Split the dataset into training and testing sets
X_train_telco, X_test_telco, y_train_telco, y_test_telco = train_test_split(X_telco_scaled, y_telco, test_size=0.2, random_state=42)

# Train logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_telco, y_train_telco)

# Train k-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_telco, y_train_telco)

# Evaluate models
log_pred = log_model.predict(X_test_telco)
knn_pred = knn_model.predict(X_test_telco)

print("Logistic Regression Classification Report:")
print(classification_report(y_test_telco, log_pred))

print("K-NN Classification Report:")
print(classification_report(y_test_telco, knn_pred))

# Compute confusion matrices
log_cm = confusion_matrix(y_test_telco, log_pred)
print("Logistic Regression Confusion Matrix:")
print(log_cm)

knn_cm = confusion_matrix(y_test_telco, knn_pred)
print("K-NN Confusion Matrix:")
print(knn_cm)