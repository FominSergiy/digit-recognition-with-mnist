#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
import os


path = os.path.dirname(os.path.abspath(__file__))
training_images_filepath = f'{path}/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = f'{path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath =  f'{path}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath =  f'{path}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

#
# MNIST Data Loader Class
#
class MnistDataloader:
    def __init__(self):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        # modifier to get a format similar to tensor flow thing? hm
        x_train = np.stack(x_train)
        y_train = np.stack(y_train)
        x_test = np.stack(x_test)
        y_test = np.stack(y_test)

        return (x_train, y_train),(x_test, y_test)    