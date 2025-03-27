import numpy as np
import matplotlib.pyplot as plt

def hebbian_learning(X, y):
    weights = np.zeros(X.shape[1])
    weight_updates = [weights.copy()]  # Track weight updates

    for i in range(len(y)):
        weights += X[i] * y[i]
        weight_updates.append(weights.copy())  # Save update

    return weights, np.array(weight_updates)

# Training Data
X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
y = np.array([1, 1, -1, -1])

# Compute weight updates
weights, hebbian_updates = hebbian_learning(X, y)
print("Final Weights:", weights)

# Plot Hebbian Learning Weight Updates
plt.plot(hebbian_updates[:, 0], marker='o', label="Weight 1")
plt.plot(hebbian_updates[:, 1], marker='s', label="Weight 2")
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.title("Hebbian Learning Weight Updates")
plt.legend()
plt.grid()
plt.show()
