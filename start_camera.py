import time
import numpy as np
from picamera2 import Picamera2, Preview
from PIL import Image, ImageOps
import tensorflow as tf
from const import FILE_NAME
import os 
import uuid
import cv2

print('\n LOADING MODEL...')
model = tf.keras.models.load_model(FILE_NAME)
print('MODEL LOADED.\n')

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding (creates a binary image)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # thinken lines to make 1s detectable
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours to isolate digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assumes it's the digit)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the digit
        digit = thresh[y:y+h, x:x+w]
    else:
        # fallback if no contours
        digit = thresh

    # Resize to 20x20, then pad to 28x28 and center
    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Convert to float and normalize
    normalized = padded.astype(np.float32) / 255.0
    normalized = normalized.reshape(1, 28, 28, 1)

    # For saving image
    pil_img = Image.fromarray((padded).astype(np.uint8))

    return normalized, pil_img

def save_processed_img(img, predicted_num):
    dir = "captured_images"

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created directory at path: {os.path(dir)}")

    rand_name = f"{predicted_num}_{uuid.uuid4().hex}_.png"
    img.save(os.path.join(dir, rand_name))
    return

def process_request(request):
        frame = request.make_array("main")
        img_array, img = preprocess_frame(frame)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        if confidence > 0.95:
            print(f"Predicted digit: {digit} (confidence: {confidence:.2f})")
            # save_processed_img(img, digit)
        else:
            print("No digit detected")

picam2 = Picamera2()
picam2.set_controls({"Contrast": 2.0, "Sharpness": 2.0 })
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
