import numpy as np
class Neuron:
    def __init__(self, n):
        self.w = np.random.rand(n)
    def a(self, i):
        return np.dot(i, self.w)
    def l(self, i, r):
        self.w += r * self.a(i) * i

n = Neuron(3)
i = np.array([0.5, 0.3, 0.2])
for _ in range(1000):
    n.l(i, 0.1)
print("Learned weights:", n.w)