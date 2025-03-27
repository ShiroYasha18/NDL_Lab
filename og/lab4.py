import numpy as np


# Step function (for perceptron output)
def step_function(x):
    return np.where(x >= 0, 1, 0)


# Training data (X) and labels (y)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND Gate labels

# Initialize weights and bias
weights = np.random.randn(2) * 0.01  # Small random values
bias = np.random.randn() * 0.01
learning_rate = 0.1
epochs = 10

# Perceptron Learning Algorithm
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(X[i], weights) + bias
        y_pred = step_function(z)  # Apply activation function

        # Compute error
        error = y[i] - y_pred
        total_error += abs(error)

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

    print(f"Epoch {epoch + 1} - Total Error: {total_error}")

# Final predictions
predictions = step_function(np.dot(X, weights) + bias)
print("\nFinal Predictions:", predictions)
print("Correct Predictions:", np.array_equal(predictions, y))
