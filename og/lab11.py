import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

# Load data
X, y = load_digits(return_X_y=True)
X = X.reshape(-1, 8, 8, 1).astype('float32') / 16
y = np.eye(10)[y]  # One-hot encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Architectures
def create_lenet():
    model = Sequential([
        Conv2D(6, (3,3), activation='relu', input_shape=(8,8,1)),
        AveragePooling2D((2,2)),
        Conv2D(16, (3,3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_alexnet():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(8,8,1)),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_vgg():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(8,8,1)),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_placesnet():
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(8,8,1)),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train and compare
models = {
    'LeNet': create_lenet(),
    'AlexNet': create_alexnet(),
    'VGG': create_vgg(),
    'PlacesNet': create_placesnet()
}

for name, model in models.items():
    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=0)
    print(f"{name} test accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")

# Visualize predictions
plt.figure(figsize=(12,3))
for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(1,4,i)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title(f"{name}\nPred: {np.argmax(model.predict(X_test[i:i+1]))}")
    plt.axis('off')
plt.tight_layout()
plt.show()