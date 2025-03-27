import numpy as np
from tensorflow.python import keras

# Perceptron Learning for AND Gate
def perceptron(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        for i in range(len(y)):
            y_pred = np.dot(X[i], weights) + bias
            y_pred = 1 if y_pred > 0 else 0
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error
    return weights, bias

# AND Gate Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = perceptron(X, y)
print("Final Weights:", weights)
print("Final Bias:", bias)


# XOR using Multi-Layer Perceptron
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Define MLP Model
model = keras.Sequential([
    keras.layers.Dense(2, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# Test Predictions
print("Predictions:", model.predict(X).round())
