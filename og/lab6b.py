import numpy as np


# Sigmoid function & derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))


def dsigmoid(x): return x * (1 - x)


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1, time_steps=3):
        self.lr, self.time_steps = lr, time_steps
        self.Wxh, self.Whh, self.Why = np.random.randn(hidden_dim, input_dim), np.random.randn(hidden_dim,
                                                                                               hidden_dim), np.random.randn(
            output_dim, hidden_dim)
        self.h_prev = np.zeros((hidden_dim, 1))

    def forward(self, X):
        h, y = [], []
        h_t = self.h_prev
        for t in range(self.time_steps):
            h_t = sigmoid(self.Wxh @ X[t] + self.Whh @ h_t)
            y.append(sigmoid(self.Why @ h_t))
            h.append(h_t)
        self.h_prev = h_t
        return np.array(h), np.array(y)

    def backward(self, X, Y, h, y):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dh_next = np.zeros_like(self.h_prev)

        for t in reversed(range(self.time_steps)):
            dy = (y[t] - Y[t]) * dsigmoid(y[t])
            dWhy += dy @ h[t].T
            dh = self.Why.T @ dy + dh_next
            dh_raw = dh * dsigmoid(h[t])
            dWxh += dh_raw @ X[t].T
            dWhh += dh_raw @ (h[t - 1] if t > 0 else self.h_prev).T
            dh_next = self.Whh.T @ dh_raw

        # Update weights correctly
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy

    def train(self, X, Y, epochs=500):
        for epoch in range(epochs):
            h, y = self.forward(X)
            self.backward(X, Y, h, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean((y - Y) ** 2):.4f}")


# Sample sequential data
np.random.seed(42)
X_train, Y_train = np.random.rand(3, 2, 1), np.array([[0.5], [0.7], [0.2]])

# Train RNN
rnn = RNN(input_dim=2, hidden_dim=4, output_dim=1)
rnn.train(X_train, Y_train)
