import tensorflow as tf
import numpy as np
from PIL import Image
from mnist_dataset import MnistDataloader
import time

# Load data
data_loader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = data_loader.load_data()

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Network parameters
learning_rate = 1e-4
n_iterations = 50  # Changed to 50 epochs
batch_size = 128
dropout = 0.5

# Create the model using Keras Sequential API
model = tf.keras.Sequential([
    # tf.keras.Input((28, 28)),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create a callback to print progress
class PrintProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Iteration {epoch}\t| Loss = {logs['loss']:.5f}\t| Accuracy = {logs['accuracy']:.5f}")

# Train the model with timing
start_time = time.time()
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=n_iterations,
    verbose=0,
    callbacks=[PrintProgressCallback()]
)
end_time = time.time()

# Calculate and print timing information
total_time = end_time - start_time
time_per_epoch = total_time / n_iterations
total_time_minutes = total_time / 60

print(f"\nTiming Information:")
print(f"Time per epoch: {time_per_epoch:.2f} seconds")
print(f"Total training time: {total_time_minutes:.2f} minutes")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAccuracy on test set: {test_accuracy:.4f}")

file_name = 'digit-recognition.keras'
model.save(f"trained_model/{file_name}")
print(f'model saved to: {file_name}')