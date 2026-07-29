import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

x = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# Fit linear regression
model = LinearRegression()
model.fit(x, y)

# Output coefficient
print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(x, y))

# Plot regression
sns.scatterplot(x=df['total_bill'], y=df['tip'], color="blue")
plt.plot(df['total_bill'], model.predict(x), color="red", label="Regression Line")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Tip vs Total Bill")
plt.legend()
plt.show()
