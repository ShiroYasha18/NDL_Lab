import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense

# Generating a toy dataset (sequential data)
X_seq = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])  # Input sequences
y_seq = np.array([3, 4, 5])  # Expected output

# Reshape input to match RNN's expected 3D shape (samples, timesteps, features)
X_seq = X_seq.reshape((3, 3, 1))

# Define RNN model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),  # RNN layer
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_seq, y_seq, epochs=100, verbose=0)

# Make predictions
predictions = model.predict(X_seq)
print("\nPredictions:", predictions.flatten())
