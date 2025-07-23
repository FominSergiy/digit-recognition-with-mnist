import tensorflow as tf
import numpy as np
from PIL import Image
import os
from const import FILE_NAME


if not os.path.exists(FILE_NAME):
    print(f"Model file '{FILE_NAME}' not found. Please run train.py to train and save the model.")
else:
    # Load the trained model
    model = tf.keras.models.load_model(FILE_NAME)

    try:
        # Predict on own image
        img = np.invert(Image.open("test_img.png").convert('L'))
        img = img.astype('float32') / 255.0
        prediction = model.predict(np.expand_dims(img, axis=0))
        print(f"Prediction for test image: {np.argmax(prediction)}")
    except FileNotFoundError:
        print("Error: test_img.png not found.")