import numpy as np

# Bipolar sign function
def sign(x): return 1 if x >= 0 else -1

# Stored patterns
patterns = np.array([[1, -1, 1], [-1, 1, -1]])

# Weight matrix
W = np.zeros((3, 3))
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

# Initial state to recall
x = np.array([1, -1, -1])

# Async update until convergence
prev_x = None
while not np.array_equal(x, prev_x):
    prev_x = x.copy()
    for i in range(len(x)):
        x[i] = sign(W[i] @ x)

print("Recalled:", x)
