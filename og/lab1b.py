import numpy as np

# Perceptron Learning Algorithm
def perceptron_train(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(y)):
            y_pred = np.dot(X[i], weights) + bias
            y_pred = 1 if y_pred > 0 else 0
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error
    return weights, bias

# AND Gate Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = perceptron_train(X, y)
print("Final Weights:", weights)
print("Final Bias:", bias)
