import time
import numpy as np
from picamera2 import Picamera2, Preview
from PIL import Image
import tensorflow as tf
from const import FILE_NAME
import os 
import uuid

print('\n LOADING MODEL...')
model = tf.keras.models.load_model(FILE_NAME)
print('MODEL LOADED.\n')

def preprocess_frame(frame):
   # print("original shape: ", frame.shape)
    img = Image.fromarray(frame)
    img = img.convert('L')  # Grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
   # print("after resize: ", img_array.shape)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
   # print("final shape: ", img_array.shape)
    return img_array, img

def save_processed_img(img, predicted_num):
    dir = "images"
    rand_name = f"{uuid.uuid4().hex}_{predicted_num}.png"
    img.save(os.path.join(dir, rand_name))
    return

def process_request(request):
        frame = request.make_array("main")
        img_array, img = preprocess_frame(frame)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        if confidence > 0.7:
            print(f"Predicted digit: {digit} (confidence: {confidence:.2f})")
            save_processed_img(img, digit)
        else:
            print("No digit detected")

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores" )
picam2.configure(config)
picam2.start_preview(Preview.QTGL)
picam2.post_callback = process_request
picam2.start()

try:
    while True:
        time.sleep(2)
except KeyboardInterrupt:
    picam2.stop_preview()
    picam2.stop()
    print("Stopped.")
