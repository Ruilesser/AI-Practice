import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Define features and target variables
features = df[['total_bill', 'size']]
target = df['tip']

print("Features: \n", features.head())
print("Target: \n", target.head())

# Want to use 20% for testing and 80% for training
# random_state is seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training Data Set: ", X_train.shape)
print("Testing Data Set: ", X_test.shape)

# Visualize the training data
sns.pairplot(df, x_vars=['total_bill', 'size'], y_vars='tip', height=5, aspect=0.7, kind="scatter")
plt.title("Feature vs Target")
plt.show()