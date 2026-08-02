import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

# Plot
plt.plot(z, sigmoid_values)
plt.xlabel('z')
plt.ylabel('Sigma(z)')
plt.title('Sigmoid Function')
plt.grid()
plt.show()