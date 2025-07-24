import time
import numpy as np
from picamera2 import Picamera2
from PIL import Image
import tensorflow as tf
from const import FILE_NAME

print('\n LOADING MODEL...')
model = tf.keras.models.load_model(FILE_NAME)
print('MODEL LOADED.\n')

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (28, 28)})
picam2.configure(config)
picam2.start()

def preprocess_frame(frame):
    img = Image.fromarray(frame)
    img = img.convert('L')  # Grayscale
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

try:
    while True:
        frame = picam2.capture_array()
        img_array = preprocess_frame(frame)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        if confidence > 0.7:
            print(f"Predicted digit: {digit} (confidence: {confidence:.2f})")
        else:
            print("No digit detected")
        time.sleep(0.1)
except KeyboardInterrupt:
    picam2.stop()
    print("Stopped.")
    print("Stopped.")
