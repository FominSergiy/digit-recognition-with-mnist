import time
import numpy as np
from picamzero import PicamZero
from PIL import Image
import tensorflow as tf  # or import torch if using PyTorch
from const import FILE_NAME

# Load your pre-trained model (adjust path and loading as needed)
model = tf.keras.models.load_model(FILE_NAME)

# Initialize camera
cam = PicamZero()
cam.start_preview()  # Optional: shows preview window

def preprocess_frame(frame):
    # Convert frame (numpy array) to PIL Image for resizing
    img = Image.fromarray(frame)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize if your model expects it
    img_array = img_array.reshape(1, 28, 28, 1)  # Adjust shape for model
    return img_array

try:
    while True:
        frame = cam.capture_array()  # Get frame as numpy array
        img_array = preprocess_frame(frame)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        if confidence > 0.7:  # Adjust threshold as needed
            print(f"Predicted digit: {digit} (confidence: {confidence:.2f})")
        else:
            print("No digit detected")
        time.sleep(0.1)  # Adjust frame rate as needed

except KeyboardInterrupt:
    cam.stop_preview()
    print("Stopped.")
