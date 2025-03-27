import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, input_dim, grid_size, lr=0.1, sigma=1.0, epochs=100):
        self.grid_size = grid_size
        self.lr = lr
        self.sigma = sigma
        self.epochs = epochs
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)  # Random initialization
        self.neuron_positions = np.array([[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])])

    def find_bmu(self, x):
        """Find the Best Matching Unit (BMU)."""
        distances = np.linalg.norm(self.weights - x, axis=2)  # Compute distance to all neurons
        return np.unravel_index(np.argmin(distances), self.grid_size)  # BMU position

    def update_weights(self, x, bmu, iteration):
        """Update weights using a Gaussian neighborhood function."""
        lr_t = self.lr * np.exp(-iteration / self.epochs)  # Decaying learning rate
        sigma_t = self.sigma * np.exp(-iteration / self.epochs)  # Decaying neighborhood size

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                influence = np.exp(-distance_to_bmu**2 / (2 * sigma_t**2))  # Gaussian function
                self.weights[i, j] += lr_t * influence * (x - self.weights[i, j])  # Update rule

    def train(self, X):
        """Train the SOM using competitive learning."""
        for epoch in range(self.epochs):
            np.random.shuffle(X)  # Shuffle input samples
            for x in X:
                bmu = self.find_bmu(x)  # Find best matching unit
                self.update_weights(x, bmu, epoch)  # Update weights

    def visualize(self):
        """Visualize the trained feature map."""
        plt.imshow(self.weights.reshape(self.grid_size[0], self.grid_size[1], -1))
        plt.title("Self-Organizing Map Feature Map")
        plt.show()

# Generate Random 3D Input Data (for visualization)
data = np.random.rand(500, 3)  # 500 samples, 3 features

# Train a 10x10 SOM
som = SelfOrganizingMap(input_dim=3, grid_size=(10, 10), epochs=100)
som.train(data)
som.visualize()
