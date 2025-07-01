# Task 2: Deep Learning with TensorFlow/Keras
# Dataset: MNIST Handwritten Digits
# Goal: Build a CNN model to classify handwritten digits, achieve >95% test accuracy,
#       and visualize model's predictions on 5 sample images.

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print("Starting MNIST Handwritten Digits Classification Task...")

# 1. Load the MNIST Dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"MNIST dataset loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")

# 2. Preprocess the data

# Reshape the data to add a channel dimension (for CNN input)
# MNIST images are 28x28 grayscale, so channel is 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the target variable
num_classes = 10  # Digits 0-9
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Data preprocessing complete: Reshaped and normalized images, one-hot encoded labels.")

# 3. Build the CNN Model
print("Building CNN model architecture...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5), # Add dropout for regularization
    
    Dense(num_classes, activation='softmax') # Output layer for 10 classes
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()
print("CNN model built and compiled.")

# 4. Train the model
print("Training the CNN model...")
history = model.fit(
    X_train, y_train, 
    epochs=10, # You can adjust this for more training
    batch_size=128, 
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)

print("CNN model training complete.")

# 5. Evaluate the model
print("Evaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

if accuracy > 0.95:
    print("\nGoal achieved: Test accuracy > 95%!")
    # Save the trained model
    model_save_path = "mnist_cnn_model.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print("\nGoal not yet achieved: Test accuracy is not > 95%. Consider more epochs or model tuning.")

# 6. Visualize the model's predictions on 5 sample images
print("\nVisualizing predictions on 5 sample images...")

# Get 5 random indices from the test set
random_indices = np.random.choice(len(X_test), 5, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    img = X_test[idx]
    true_label = np.argmax(y_test[idx])
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)
    predicted_label = np.argmax(prediction)
    
    plt.subplot(1, 5, i + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_label}\nPred: {predicted_label}", 
              color='green' if true_label == predicted_label else 'red')
    plt.axis('off')

plt.suptitle("MNIST Predictions (True vs. Predicted Labels)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("MNIST Handwritten Digits Classification Task completed.") 