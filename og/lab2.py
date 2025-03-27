import numpy as np
import matplotlib.pyplot as plt

def hebbian_learning(X, y):
    weights = np.zeros(X.shape[1])
    for i in range(len(y)):
        weights += X[i] * y[i]
    return weights

X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
y = np.array([1, 1, -1, -1])


y = y[:, np.newaxis]  # Reshape y to (4,1) for broadcasting

# Compute weight updates
weights = hebbian_learning(X, y.flatten())  # Use y.flatten() to keep it 1D
print("Final Weights:", weights)

# Track weight updates for visualization
hebbian_updates = np.cumsum(X * y, axis=0)  # Cumulative sum for plotting

# Plot Hebbian Learning Weight Updates
plt.plot(hebbian_updates, marker='o', label=["Weight 1", "Weight 2"])
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.title("Hebbian Learning Weight Updates")
plt.legend()
plt.grid()
plt.show()
