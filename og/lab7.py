import numpy as np

def sign(x): return np.where(x >= 0, 1, -1)

patterns = np.array([[1, -1, 1], [-1, 1, -1]])
W = np.zeros((3, 3))
for p in patterns: W += np.outer(p, p)
np.fill_diagonal(W, 0)

test = np.array([1, -1, -1])
output = sign(W @ test)
print("Recalled:", output)
