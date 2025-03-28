import numpy as np

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

class MLP:
    def __init__(self):
        self.W1 = np.random.randn(2, 2)  # Hidden layer weights
        self.W2 = np.random.randn(2)  # Output layer weights
        self.b1 = np.random.randn(2)  # Hidden layer bias
        self.b2 = np.random.randn()  # Output layer bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, lr, epochs):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                h = self.sigmoid(X[i] @ self.W1 + self.b1)
                o = self.sigmoid(h @ self.W2 + self.b2)

                # Compute error
                e = y[i] - o
                total_error += e**2

                # Backpropagation
                dW2 = lr * e * o * (1 - o) * h
                dW1 = lr * e * o * (1 - o) * np.outer(X[i], self.W2 * h * (1 - h))

                self.W2 += dW2
                self.b2 += lr * e * o * (1 - o)
                self.W1 += dW1
                self.b1 += lr * e * o * (1 - o) * self.W2 * h * (1 - h)

            # Print error every few epochs
            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}, LR={lr}, Error={total_error:.4f}")

# Train MLP with different learning rates
for lr in [0.1, 0.5, 1.0]:
    mlp = MLP()
    mlp.train(lr, epochs=10000)
    predictions = [round(mlp.sigmoid(mlp.sigmoid(x @ mlp.W1 + mlp.b1) @ mlp.W2 + mlp.b2)) for x in X]
    print(f"LR={lr}: {predictions}")
