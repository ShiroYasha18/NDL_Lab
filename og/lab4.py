import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# RBF transformation using input points as centers
def rbf_transform(X, centers, sigma=0.5):
    return np.exp(-cdist(X, centers) ** 2 / (2 * sigma ** 2))

X_rbf = rbf_transform(X, X)  # Transform inputs using RBF centers

plt.figure()
for reg in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=reg, fit_intercept=False).fit(X_rbf, y)
    plt.plot(model.predict(X_rbf), 'o-', label=f"λ={reg}")

plt.legend(), plt.grid(), plt.show()
