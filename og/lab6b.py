import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X_train = np.array([[[0], [1], [2]],
                     [[1], [2], [3]],
                     [[2], [3], [4]],
                     [[3], [4], [5]],
                     [[4], [5], [6]],
                     [[5], [6], [7]]]).astype(np.float32)

y_train = np.array([[3], [4], [5], [6], [7], [8]]).astype(np.float32)
model = Sequential([
    SimpleRNN(20, activation='tanh', input_shape=(3, 1)),  # More neurons
    Dense(10, activation='relu'),  # Extra dense layer to capture patterns
    Dense(1, activation='linear')  # Output layer
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=500, verbose=0)  # Increased epochs for better learning
test_sample = np.array([[[4], [5], [6]]]).astype(np.float32)
pred = model.predict(test_sample)
print("Prediction for [4, 5, 6]:", pred)