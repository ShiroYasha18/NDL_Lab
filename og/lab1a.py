import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Generate values for plotting
x = np.linspace(-5, 5, 100)

# Plot the activation functions
plt.figure(figsize=(8,6))
plt.plot(x, sigmoid(x), label="Sigmoid", linestyle='dashed')
plt.plot(x, relu(x), label="ReLU", linestyle='dotted')
plt.plot(x, tanh(x), label="Tanh", linestyle='solid')
plt.legend()
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Activation Functions")
plt.grid()
plt.show()
