import sys
import numpy as np
from collections import Counter
import math

# Set the random seed for reproducibility
np.random.seed(1)

# Load the reviews dataset
with open('reviews.txt', 'r') as f:
    raw_reviews = f.readlines()

# Load the corresponding labels dataset
with open('labels.txt', 'r') as f:
    raw_labels = f.readlines()

# Tokenize each review into a set of unique words
tokens = list(map(lambda x: set(x.split(" ")), raw_reviews))

# Create a vocabulary of all unique words
vocab = set()
for sent in tokens:
    for j in sent:
        if len(j) > 0:  # Ignore empty strings
            vocab.add(j)

# Convert the vocabulary to a list for indexing
vocab = list(vocab)

# Create a mapping from words to unique indices
word2index = {word: i for i, word in enumerate(vocab)}

# Convert each review to a list of corresponding word indices
input_dset = []
for sent in tokens:
    sent_indices = [word2index[j] for j in sent if j in word2index]
    input_dset.append(sent_indices)

# Convert labels to binary targets (1 for positive, 0 for negative)
targ_dset = [1 if label == 'positive\n' else 0 for label in raw_labels]

# Set hyperparameters
alpha = 0.01  # Learning rate
iterations = 2  # Number of training iterations
hidden_size = 100  # Number of neurons in the hidden layer

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights with small random values
weight01 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1  # Input to hidden
weight12 = 0.2 * np.random.random((hidden_size, 1)) - 0.1  # Hidden to output

# Train the neural network
for iter in range(iterations):
    correct, total = 0, 0

    # Training loop (excluding last 1000 examples for testing)
    for i in range(len(input_dset) - 1000):
        x, y = input_dset[i], targ_dset[i]

        # Forward pass
        l1 = sigmoid(np.sum(weight01[x], axis=0))  # Hidden layer
        l2 = sigmoid(l1.dot(weight12))  # Output layer

        # Calculate error (difference between prediction and target)
        l2_delta = l2 - y
        l1_delta = l2_delta.dot(weight12.T)

        # Update weights using backpropagation
        weight01[x] -= l1_delta * alpha
        weight12 -= np.outer(l1, l2_delta) * alpha

        # Track training accuracy
        if np.abs(l2_delta) < 0.5:
            correct += 1
        total += 1

        # Print training progress every 10 steps
        if i % 10 == 9:
            progress = str(i / len(input_dset))
            print('Iter:{}---progress: {}.{}---Train Acc:{}---------'.format(
                iter, progress[2:4], progress[4:6], correct / float(total)
            ))

# Evaluate the model on the last 1000 examples (test set)
correct, total = 0, 0
for i in range(len(input_dset) - 1000, len(input_dset)):
    x, y = input_dset[i], targ_dset[i]

    # Forward pass on test data
    T1 = sigmoid(np.sum(weight01[x], axis=0))
    T2 = sigmoid(T1.dot(weight12))

    # Count correct predictions
    if (T2 - y) < 0.5:
        correct += 1
    total += 1

# Print final test accuracy
print('Test Acc: {}'.format(correct / float(total)))

# Function to find the most similar words to a given target word
def similar(target='beautiful'):
    target_index = word2index[target]  # Get the index of the target word
    score = Counter()

    # Calculate Euclidean distance between target word and all other words
    for word, i in word2index.items():
        diff = weight01[i] - weight01[target_index]
        sq_diff = diff * diff
        score[word] = -math.sqrt(sum(sq_diff))

    # Return the 30 most similar words
    return score.most_common(30)

# Print words most similar to 'terrible'
print(similar('terrible'))
