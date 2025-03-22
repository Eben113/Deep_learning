"""A feedforward neural network implemented using only numpy"""

# Importing necessary libraries
import numpy as np
import random
from mnist_loader import mnist_load

# Defining the Network class
class Network(object):
    def __init__(self, sizes):
        # Initialize the network structure and parameters
        self.sizes = sizes  # List containing the number of neurons in each layer

        # Initialize weights with random values (Gaussian distribution)
        self.weights = [np.random.randn(rows, cols) for rows, cols in zip(sizes[:-1], sizes[1:])]

        # Store the number of layers
        self.num_layers = len(sizes)

        # Initialize biases with random values (Gaussian distribution)
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]

    # Perform forward propagation through the network
    def feedfoward(self, a):
        for weight, bias in zip(self.weights, self.biases):
            # Apply linear transformation and activation function (sigmoid)
            a = sigmoid(np.dot(a, weight) + bias)
        return a

    # Stochastic Gradient Descent (SGD) for training the network
    def SGD(self, training_data, mini_batch_size, epoch, eta, test_data=None):
        if test_data:
            n_test = len(test_data)  # Number of test samples

        n = len(training_data)  # Number of training samples

        for epoch in range(epoch):
            # Shuffle training data for each epoch
            random.shuffle(training_data)

            # Create mini-batches from the training data
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            # Update network parameters using each mini-batch
            for batch in mini_batches:
                self.update_mini_batches(batch, eta)

            # Evaluate the network's performance after each epoch
            if test_data:
                print('epoch:{}----{}/{}||{}/{}'.format(epoch, self.evaluate(training_data), n, self.evaluate(test_data), n_test))
            else:
                print('Epoch:{}'.format(epoch))

    # Update weights and biases using backpropagation
    def update_mini_batches(self, batch, eta):
        # Initialize gradient accumulators
        w = [np.zeros(weight.shape) for weight in self.weights]
        b = [np.zeros(bias.shape) for bias in self.biases]

        # Compute gradients for each training example in the batch
        for x, y in batch:
            w_update_delta, b_update_delta = self.backprop(x, y)

            # Accumulate gradients
            w = (nw + dnw for nw, dnw in zip(w, w_update_delta))
            b = (nb + dnb for nb, dnb in zip(b, b_update_delta))

        # Update weights and biases using the averaged gradients
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, b)]

    # Perform backpropagation to compute gradients
    def backprop(self, x, y):
        # Initialize gradient holders
        update_w = [np.zeros(weight.shape) for weight in self.weights]
        update_b = [np.zeros(bias.shape) for bias in self.biases]

        # Feedforward: store activations and weighted sums (zs)
        activations = []
        activation = x
        activations.append(x)
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            # Compute weighted input
            z = np.dot(activation, weight) + bias
            zs.append(z)

            # Apply activation function
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass: compute the gradient of the cost with respect to weights and biases

        # Compute error at the output layer
        delta = self.cost_derivative(activations[-1], y) * sig_prime(zs[-1])
        update_b[-1] = delta
        update_w[-1] = np.dot(activations[-2].T, delta)

        # Propagate error backwards through the network
        for j in range(2, self.num_layers):
            sp = sig_prime(zs[-j])
            delta = np.dot(delta, self.weights[-j + 1].T) * sp
            update_b[-j] = delta
            update_w[-j] = np.dot(activations[-j - 1].T, delta)

        return (update_w, update_b)

    # Evaluate the network's performance on test data
    def evaluate(self, test_data):
        correct_count = 0

        for x, y in test_data:
            # Predict the output and compare with the correct label
            pred = np.argmax(self.feedfoward(x))
            if pred == np.argmax(y):
                correct_count += 1

        return correct_count

    # Compute the derivative of the cost function
    def cost_derivative(self, pred, y):
        return (pred - y)

# Sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative of the sigmoid function
def sig_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Load MNIST dataset
train = mnist_load('mnist_train.csv', 10000)
test = mnist_load('mnist_test.csv', 500)

# Create a neural network with 784 input neurons, one hidden layer of 30 neurons, and 10 output neurons
net = Network((784, 30, 10))

# Train the network using stochastic gradient descent
net.SGD(train, 10, 30, 3.0, test)
