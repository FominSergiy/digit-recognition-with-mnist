# digit-recognition-with-mnist
use digit recognition neural network with camera module on raspberry pi to recognize hand-written digits


## Requirements

- Some computer that can run python3
- Raspberry Pi with (Pi Camera Module)[https://www.raspberrypi.com/products/camera-module-v2/]

## Installation
- Install Python3 and pip
- Install the Python dependencies `pip install -r requirements.txt`
- Create local venv `python3 -m venv venv`
- Unzip mnist dataset `unzip mnist_dataset/mnist-dataset.zip -d mnist_dataset `
    - I included the dataset in case the original url I used is no longer valid.
    - `source ./venv/bin/activate`
    - run the `test_run.py` to confirm mnist data loader works as expected
- Run the file `python main.py`

## Model Training

`train.py` does the model training based on mnist_dataset training data. Once the model is trained it is safed into a dir that we are going to use to `scp` to our raspberry pi host.

running `python3 train.py` trains the model

