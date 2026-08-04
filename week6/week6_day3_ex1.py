import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display information
print("Dataset Info:")
print(df.info())

# Preview the first few rows of the dataset
print("\nDataset Head:")
print(df.head())

# Apply one-hot encoding to categorical features
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display encoded dataset information
print("\n One-Hot Encoded Dataset Info:")
print(df_one_hot.info())

# Apply label encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display label encoded dataset information
print("\nLabel Encoded Dataset Info:")
print(df[['Pclass', 'Pclass_encoded']].head())

# Apply frequency encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display frequency encoded dataset information
print("\nFrequency Encoded Dataset Info:")
print(df[['Ticket', 'Ticket_frequency']].head())

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")