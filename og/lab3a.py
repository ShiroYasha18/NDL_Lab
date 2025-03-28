''' Implement the Gate Operations (AND, OR, XOR, NAND, NOR,
XNOR, NOT) using Single Layer Perceptron.'''

import numpy as np

class Perceptron:
    def __init__(self, n):
        self.w, self.b = np.random.rand(n), np.random.rand()

    def activate(self, x):
        return 1 if np.dot(x, self.w) + self.b > 0 else 0

    def train(self, X, y, lr=0.1, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                self.w += lr * (yi - self.activate(xi)) * xi
                self.b += lr * (yi - self.activate(xi))

logic_gates = {
    "AND":   ([[0,0], [0,1], [1,0], [1,1]], [0, 0, 0, 1]),
    "OR":    ([[0,0], [0,1], [1,0], [1,1]], [0, 1, 1, 1]),
    "NAND":  ([[0,0], [0,1], [1,0], [1,1]], [1, 1, 1, 0]),
    "NOR":   ([[0,0], [0,1], [1,0], [1,1]], [1, 0, 0, 0]),
    "NOT":   ([[0], [1]], [1, 0])
}

for gate, (X, y) in logic_gates.items():
    p = Perceptron(len(X[0]))
    p.train(np.array(X), np.array(y))
    print(f"{gate}: {[p.activate(x) for x in X]}")