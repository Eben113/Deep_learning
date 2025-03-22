# Import necessary libraries
import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset

# Load training and test datasets from local storage
train = datasets.load_from_disk('/storage/emulated/0/python/train')
test = datasets.load_from_disk('/storage/emulated/0/python/test')

# Prepare input arrays for training and testing
a = []
b = []

# Process the first 1000 images from the training dataset
for i in range(1000):
    a.append(np.array(train[i]['image']).reshape(1, 784))

# Normalize pixel values to the range [0, 1]
a = (np.concatenate(a)) / 255

# Process the first 100 images from the test dataset
for i in range(100):
    b.append(np.array(test[i]['image']).reshape(1, 784))

# Normalize pixel values to the range [0, 1]
b = (np.concatenate(b)) / 255

# Extract labels for training and testing
labels = np.array(train[0:1000]['label'])
test_labels = np.array(test[0:100]['label'])

# Initialize one-hot encoded labels for training and testing
label = np.zeros((1000, 10))
test_label = np.zeros((100, 10))

# Convert training labels to one-hot encoding
for i, l in enumerate(labels.flat):
    label[i][l] = 1

# Convert test labels to one-hot encoding
for i, l in enumerate(test_labels.flat):
    test_label[i][l] = 1

# Set random seed for reproducibility
np.random.seed(1)

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return (1 - (output) ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

# Define ReLU activation and its derivative
relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: (x >= 0)

# Hyperparameters
alpha = 2                  # Learning rate
batch_size = 128           # Batch size for training
in_rows, in_cols = 28, 28  # Input image dimensions
kernel_rows, kernel_cols = 3, 3  # Kernel dimensions
no_kernels = 16            # Number of convolutional kernels

# Calculate the size of the hidden layer after convolution
hidden_size = ((in_rows - kernel_rows) * (in_cols - kernel_cols) * no_kernels)

# Initialize kernels and weights with small random values
kernels = 0.02 * (np.random.random((kernel_rows * kernel_cols, no_kernels))) - 0.01
weight12 = 0.2 * (np.random.random((hidden_size, 10))) - 0.1

# Function to extract image sections (for convolution operation)
def get_img_sect(layer, r_from, r_to, c_from, c_to):
    sub_sect = layer[:, r_from:r_to, c_from:c_to]
    return sub_sect.reshape(-1, 1, r_to - r_from, c_to - c_from)

# Training loop (350 epochs)
for i in range(350):
    error, correct_cnt = 0.0, 0

    # Iterate over mini-batches
    for j in range(int(len(a) / batch_size)):
        batch_start, batch_end = (j * batch_size), ((j + 1) * batch_size)

        # Prepare input batch and reshape for convolution
        l0 = a[batch_start:batch_end]
        l0 = l0.reshape(l0.shape[0], 28, 28)

        # Perform convolution: extract patches from input images
        sects = []
        for r_start in range(in_rows - kernel_rows):
            for col_start in range(in_cols - kernel_cols):
                sect = get_img_sect(l0, r_start, r_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)

        # Flatten and concatenate image sections
        expanded_in = np.concatenate(sects, axis=1)
        es = expanded_in.shape
        flat_in = expanded_in.reshape(es[0] * es[1], -1)

        # Apply convolution (dot product with kernels)
        kernel_out = np.dot(flat_in, kernels)

        # Apply activation function (tanh)
        l1 = tanh(kernel_out.reshape(es[0], -1))

        # Apply dropout for regularization
        dropout_mask = np.random.randint(2, size=l1.shape)
        l1 *= dropout_mask * 2

        # Compute softmax output
        l2 = softmax(np.dot(l1, weight12))

        # Count correct predictions
        for k in range(batch_size):
            correct_cnt += int(np.argmax(l2[k:k + 1]) == np.argmax(label[batch_start + k:batch_start + k + 1]))

        # Compute gradients (backpropagation)
        delta12 = (label[batch_start:batch_end] - l2) / (batch_size * l2.shape[0])
        delta01 = (delta12.dot(weight12.T)) * tanh2deriv(l1)
        delta01 *= dropout_mask

        # Update weights using gradients
        weight12 += alpha * (l1.T.dot(delta12))

        # Reshape deltas for kernel updates
        l1d_reshape = delta01.reshape(kernel_out.shape)
        k_update = flat_in.T.dot(l1d_reshape)

        # Update kernels
        kernels += alpha * k_update

    # Evaluate model on the test dataset
    test_correct_cnt = 0
    for j in range(len(b)):
        l0 = b[j:j + 1].reshape(1, 28, 28)

        sects = []
        for r_start in range(in_rows - kernel_rows):
            for col_start in range(in_cols - kernel_cols):
                sect = get_img_sect(l0, r_start, r_start + kernel_rows, col_start, col_start + kernel_cols)
                sects.append(sect)

        expanded_in = np.concatenate(sects, axis=1)
        es = expanded_in.shape
        flat_in = expanded_in.reshape(es[0] * es[1], -1)

        kernel_out = np.dot(flat_in, kernels)
        l1 = tanh(kernel_out.reshape(es[0], -1))
        l2 = np.dot(l1, weight12)

        test_correct_cnt += int(np.argmax(l2) == np.argmax(test_label[j:j + 1]))

    # Print training and test accuracy every 10 epochs
    if i % 10 == 0 or i == 349:
        print('i: {}'.format(i))
        print('    Correct: {:.4f}'.format(correct_cnt / float(len(a))))
        print('    test_Correct: {:.4f}'.format(test_correct_cnt / len(b)))
