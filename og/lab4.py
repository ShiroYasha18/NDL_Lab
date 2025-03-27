import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist

# XOR Input and Output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR labels

# Define RBF function (Gaussian Kernel)
def rbf(x, centers, sigma=1.0):
    return np.exp(-cdist(x, centers) ** 2 / (2 * sigma ** 2))

# RBF Centers (Use Input Data as Centers)
centers = X.copy()
sigma = 0.5  # Fixed spread parameter

# Transform Input using RBF
X_rbf = rbf(X, centers, sigma)

# Experiment with different regularization values
lambdas = [0, 0.01, 0.1, 1, 10]  # Regularization parameters
weights_list = []

plt.figure(figsize=(8, 5))

for reg in lambdas:
    # Train using Ridge Regression
    model = Ridge(alpha=reg, fit_intercept=False)  # Regularization controlled by alpha
    model.fit(X_rbf, y)
    weights_list.append(model.coef_)

    # Predicted Outputs
    y_pred = model.predict(X_rbf)

    # Plot Results
    plt.plot(y_pred, marker='o', label=f"λ={reg}")

plt.xlabel("XOR Sample Index")
plt.ylabel("Predicted Output")
plt.title("Effect of Regularization on RBF Network for XOR")
plt.legend()
plt.grid()
plt.show()

# Print final weights for analysis
for i, reg in enumerate(lambdas):
    print(f"λ={reg}, Weights: {weights_list[i]}")
