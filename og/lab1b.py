import numpy as np
def perceptron(X, y, lr=0.1, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        for i in range(len(y)):
            pred = 1 if np.dot(X[i], w) + b > 0 else 0
            error = y[i] - pred
            w += lr * error * X[i]
            b += lr * error
    return w, b

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])
weights, bias = perceptron(X, y)
predictions = [1 if np.dot(x, weights) + bias > 0 else 0 for x in X]
for i, x in enumerate(X):
    print(f"Input: {x}, Predicted: {predictions[i]}")
print("Weights:", weights)
print("Bias:", bias)