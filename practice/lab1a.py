import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid

x = np.linspace(-5, 5, 100)

activations = {
    "Sigmoid": expit(x),
    "ReLU": np.maximum(0, x),
    "Tanh": np.tanh(x),
    "Leaky ReLU": np.where(x > 0, x, 0.01 * x),
    "ELU": np.where(x > 0, x, 0.01 * (np.exp(x) - 1)),
    "Softplus": np.log1p(np.exp(x)),
    "Softsign": x / (1 + np.abs(x))
}

plt.figure(figsize=(10, 7))
for name, values in activations.items():
    plt.plot(x, values, label=name)

plt.legend()
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Activation Functions")
plt.grid()
plt.show()
