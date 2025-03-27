import numpy as np

class MLP:
    def __init__(self):
        self.W1, self.b1 = np.random.rand(2, 2), np.random.rand(2)
        self.W2, self.b2 = np.random.rand(2), np.random.rand()

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def dsigmoid(self, x): return x * (1 - x)

    def forward(self, x):
        self.h = self.sigmoid(np.dot(x, self.W1) + self.b1)
        return self.sigmoid(np.dot(self.h, self.W2) + self.b2)

    def train(self, X, y, lr, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                o = self.forward(X[i])
                error = y[i] - o
                dW2 = lr * error * self.dsigmoid(o) * self.h
                dW1 = lr * error * self.dsigmoid(o) * self.W2 * self.dsigmoid(self.h) * X[i][:, None]
                self.W2 += dW2
                self.b2 += lr * error * self.dsigmoid(o)
                self.W1 += dW1
                self.b1 += lr * error * self.dsigmoid(o) * self.W2

# XOR Training Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Train XOR with different learning rates
for lr in [0.1, 0.5, 1.0]:
    mlp = MLP()
    mlp.train(X, y, lr, epochs=10000)
    print(f"LR={lr}: {[round(mlp.forward(x)) for x in X]}")