import sys, random
import numpy as np
from collections import Counter
import math

# Set a random seed for reproducibility
np.random.seed(1)

# Load the raw review data from the text file
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

# Tokenize the reviews by splitting each review into words
tokens = list(map(lambda x: x.split(' '), raw_reviews))

# Create a word counter to track word frequencies
wordcnt = Counter()

# Populate the word counter with word occurrences
for token in tokens:
	for word in token:
		wordcnt[word] -= 1  # Negative count for reverse sorting (most common first)

# Create a vocabulary from the most common words
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

# Map each word to a unique index
word2index = {}
for i, word in enumerate(vocab):
	word2index[word] = i

# Prepare training data
concatenate = []  # Stores all word indices
input_dataset = []  # Stores reviews as sequences of word indices

for sent in tokens:
	sent_indices = []  # Holds the current review's word indices
	for word in sent:
		try:
			# Map each word to its index and collect
			concatenate.append(word2index[word])
			sent_indices.append(word2index[word])
		except:
			pass  # Ignore words not in the vocabulary
	input_dataset.append(sent_indices)

# Convert the concatenated list to a numpy array
concatenated = np.array(concatenate)

# Shuffle the dataset to improve training
random.shuffle(input_dataset)

# Set model hyperparameters
alpha, iterations = 0.05, 2  # Learning rate and number of training epochs
hidden_size, window, negative = 50, 2, 5  # Embedding size, context window, and negative samples

# Initialize weights with small random values
weight01 = 0.2 * np.random.rand(len(vocab), hidden_size) - 0.5  # Input-to-hidden weights
weight12 = np.random.rand(len(vocab), hidden_size)  # Hidden-to-output weights

# Create a target vector for positive and negative samples
layer2target = np.zeros(negative + 1)
layer2target[0] = 1  # Positive sample is at index 0

# Function to find the most similar words to a given word
def similar(word):
	targ_index = word2index[word]  # Get the index of the target word
	count = Counter()
	for word, index in word2index.items():
		# Compute Euclidean distance between word embeddings
		raw_difference = weight01[index] - weight01[targ_index]
		sq_diff = raw_difference * raw_difference
		dist = -np.mean(np.sqrt(sq_diff))
		count[word] = dist
	return count.most_common(10)  # Return the 10 most similar words

# Sigmoid activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Train the Skip-gram with Negative Sampling (SGNS) model
for i, review in enumerate(input_dataset * iterations):
	for targ in range(len(review)):
		# Create a target sample and negative samples
		targ_sample = [review[targ]] + list(concatenated[np.random.randint(len(concatenated), size=negative)])

		# Collect context words within the window
		left = review[max(0, targ - window):targ]
		right = review[targ + 1:min(targ + window + 1, len(review))]

		# Forward pass: compute hidden layer as the mean of context word embeddings
		l1 = np.mean(weight01[left + right], axis=0)

		# Compute output layer probabilities for positive and negative samples
		l2 = sigmoid(l1.dot(weight12[targ_sample].T))

		# Compute output layer error (delta)
		l2_delta = l2 - layer2target

		# Backpropagate the error to the hidden layer
		l1_delta = l2_delta.dot(weight12[targ_sample])

		# Update weights using gradient descent
		weight12[targ_sample] -= np.outer(l2_delta, l1) * alpha
		weight01[left + right] -= l1_delta * alpha

		# Print training progress every 250 steps
		if i % 250 == 0:
			print('progress: {:.2f}% ----
{}'.format((i / (len(input_dataset) * iterations)) * 100, similar('terrible')))

# Output the final most similar words to 'terrible'
print(similar('terrible'))
