import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA
# Generate random input data  
data = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 1000)
# Hebbian Learning  
w = np.random.randn(2)  
for x in data:  
    w += 0.01 * np.dot(w, x) * x
# PCA for comparison  
pca = PCA(n_components=1).fit(data).components_[0]
# Normalize and visualize  
w, pca = w / np.linalg.norm(w), pca / np.linalg.norm(pca)  
plt.scatter(data[:, 0], data[:, 1], alpha=0.3)  
plt.quiver(0, 0, w[0], w[1], color='r', scale=3, label="Hebbian")  
plt.quiver(0, 0, pca[0], pca[1], color='g', scale=3, label="PCA")  
plt.legend(), plt.grid(), plt.axis('equal'), plt.show()