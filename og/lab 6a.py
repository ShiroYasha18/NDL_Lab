import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Generate random data (100 samples, 3 features)
data = np.random.rand(100, 3)

# Initialize the SOM (grid size 10x10, input dim 3, sigma=2.0 for better topology)
som = MiniSom(10, 10, 3, sigma=2.0, learning_rate=0.5)

# Initialize weights using PCA for better ordering
som.random_weights_init(data)

# Train SOM with batch training (5000 iterations for better convergence)
som.train_batch(data, 5000)

# Visualize the trained SOM weights
plt.imshow(som.get_weights().reshape(10, 10, 3))
plt.title("Topologically Ordered Self-Organizing Map")
plt.show()