import numpy as np
import matplotlib.pyplot as plt

def hebbian_learning(X):
    return np.dot(X.T, X)

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

weights = hebbian_learning(X)
print("Weight Matrix (Principal Component Approximation):\n", weights)

hebbian_updates = np.cumsum(X, axis=0)

# Plot Hebbian Learning Process
plt.figure(figsize=(8,6))
plt.plot(hebbian_updates[:, 0], label="Weight 1", marker='o', linestyle='dashed')
plt.plot(hebbian_updates[:, 1], label="Weight 2", marker='s', linestyle='dotted')
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.title("Hebbian Learning Weight Updates")
plt.legend()
plt.grid()
plt.show()
