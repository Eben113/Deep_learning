# Import necessary libraries
import csv
import itertools
import numpy as np

def mnist_load(file, size):
    # Open the CSV file containing MNIST data
    file = open(file)

    # Read the file using the CSV reader
    file = csv.reader(file)

    # Generate combinations of rows from the CSV file, taking 'size + 1' rows
    random = itertools.combinations(file, size + 1)

    # Get the next combination (first 'size + 1' rows)
    random = next(random)

    # Initialize an empty list to store the processed output
    output = []

    # Iterate over the desired number of samples
    for i in range(1, size + 1):

        # Extract pixel data (from the second column onward), convert to an integer array, and normalize to [0, 1]
        data = np.array(random[i][1:], dtype='int64').reshape(1, -1) / 255

        # Convert the label (first column) to a one-hot encoded vector
        label = vectorize(random[i][0])

        # Append the data-label pair to the output list
        output.append((data, label))

    # Return the processed dataset
    return output

def vectorize(x):
    # Create a zero vector of shape (1, 10) for one-hot encoding
    out = np.zeros((1, 10))

    # Set the corresponding class index to 1
    out[0, int(x)] = 1

    # Return the one-hot encoded vector
    return out
