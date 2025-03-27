import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate random 2D data with Gaussian distribution
def generate_data(samples=1000):
    return np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], samples)

# Single linear neuron with Hebbian learning
class HebbianNeuron:
    def __init__(self, input_dim, lr=0.01):
        self.w = np.random.randn(input_dim)
        self.lr = lr

    def train(self, data, epochs=10):
        for _ in range(epochs):
            for x in data:
                self.w += self.lr * np.dot(self.w, x) * x  # Hebbian update

# Train neuron & extract PCA
data = generate_data()
neuron = HebbianNeuron(input_dim=2)
neuron.train(data)

pca = PCA(n_components=1).fit(data).components_[0]

# Normalize vectors
w_norm, pca_norm = neuron.w / np.linalg.norm(neuron.w), pca / np.linalg.norm(pca)

# Plot data & principal directions
plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Data")
plt.quiver(0, 0, *w_norm, color='r', scale=3, label="Hebbian")
plt.quiver(0, 0, *pca_norm, color='g', scale=3, label="PCA")
plt.legend(), plt.grid(), plt.axis('equal')
plt.title("Hebbian Learning vs PCA")
plt.show()

print("Normalized Hebbian Weights:", w_norm)
print("Normalized Principal Component:", pca_norm)
