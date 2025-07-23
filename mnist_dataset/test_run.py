#
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt
from loader import MnistDataloader
import os


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, titles=None, rows=2, cols=5):
    """
    Display a grid of MNIST images with their labels
    Args:
        images: List of image arrays
        titles: List of titles (labels) for each image
        rows: Number of rows in the grid
        cols: Number of columns in the grid
    """
    plt.figure(figsize=(10, 4))
    for i in range(min(len(images), rows * cols)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        if titles is not None:
            plt.title(f'Label: {titles[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training \nimage [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test \nimage [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
