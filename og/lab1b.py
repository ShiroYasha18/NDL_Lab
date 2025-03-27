import numpy as np

# Perceptron Learning Algorithm
def perceptron_train(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(y)):
            y_pred = np.dot(X[i], weights) + bias
            y_pred = 1 if y_pred > 0 else 0  # Binary classification
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias

# Prediction function
def perceptron_predict(X, weights, bias):
    return [1 if np.dot(x, weights) + bias > 0 else 0 for x in X]

# AND Gate Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = perceptron_train(X, y)

# Testing the trained perceptron
predictions = perceptron_predict(X, weights, bias)
for i, x in enumerate(X):
    print(f"Input: {x}, Predicted Output: {predictions[i]}")

print("Final Weights:", weights)
print("Final Bias:", bias)
