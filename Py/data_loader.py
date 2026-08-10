import urllib.request # urllib module is used to work with URLs. urllib.request is used to make requests to websites and download files
from pathlib import Path # Path is a python class used to work with filesystem paths


#Downloading the required files locally
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/" # MNIST lives here

files = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
}

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True) # creating the directory data/raw/. parents make sure if data or raw doesn't exist, they will be created and exist_ok makes sure if those directories exist, then we don't throw an error.

for filename in files:
    path = raw_dir / filename

    if not path.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(base_url + filename, path)


# Image parser
import gzip
import struct
import numpy as np

def load_images(filename):
    with gzip.open(filename, "rb") as f:
        data = f.read()

    magic, num_images, rows, cols = struct.unpack(">IIII", data[:16])

    if magic != 2051:
        raise ValueError("Invalid MNIST image file")

    images = np.frombuffer(data[16:], dtype = np.uint8)

    images = images.reshape(num_images, rows, cols)

    return images

X_train = load_images("data/raw/train-images-idx3-ubyte.gz")
X_test = load_images("data/raw/t10k-images-idx3-ubyte.gz")

# print("Training data shape: ", X_train.shape)
# print("Testing data shape: ", X_test.shape)


# Label parser
def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        data = f.read()

    magic, num_labels = struct.unpack(">II", data[:8])

    if magic != 2049:
        raise ValueError("Invalid MNIST label files")

    labels = np.frombuffer(data[8:], dtype=np.uint8)

    return labels

y_train = load_labels("data/raw/train-labels-idx1-ubyte.gz")
y_test = load_labels("data/raw/t10k-labels-idx1-ubyte.gz")

# print("Training truth shape: ", y_train.shape)
# print("Testing truth shape: ", y_test.shape)


# Data pre-processing

## Flattening the images
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# print("Training data new shape: ", X_train.shape)
# print("Testing data new shape: ", X_test.shape)

## Combining labels and pixels: We will stack y on top of X
train_data = np.column_stack((y_train, X_train))
test_data = np.column_stack((y_test, X_test))

print("Training truth shape: ", train_data.shape)
print("Testing truth shape: ", test_data.shape)


# Saving as CSV files
np.savetxt("data/mnist_train.csv", train_data, delimiter=",", fmt="%d")
np.savetxt("data/mnist_test.csv", test_data, delimiter=",", fmt="%d")