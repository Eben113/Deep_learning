import numpy as np
import random
from mnist_loader import mnist_load
class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn(rows, cols) for rows,cols in  zip(sizes[:-1], sizes[1:])]
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(1,y) for y in sizes[1:]]
        
    def feedfoward(self,a):
        for weight, bias in zip(self.weights, self.biases):
            a = sigmoid(np.dot(a, weight) + bias)
        return a
    
    def SGD(self, training_data, mini_batch_size, epoch, eta, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epoch):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]for k in range(0,n,mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batches(batch, eta)
            if test_data:
                print('epoch:{}----{}/{}||{}/{}'.format(epoch,self.evaluate(training_data),n, self.evaluate(test_data), n_test))
            else:
                print('Epoch:{}'.format(epoch))
    
    def update_mini_batches(self, batch, eta):
            w = [np.zeros(weight.shape) for weight in self.weights]
            b = [np.zeros(bias.shape) for bias in self.biases]
            for x,y in batch:
                w_update_delta, b_update_delta = self.backprop(x,y)
                w = (nw+dnw for nw,dnw in zip(w, w_update_delta))
                b = (nb+dnb for nb,dnb in zip(b, b_update_delta))
            self.weights = [w-(eta/len(batch))*nw for w,nw in zip(self.weights, w)]
            self.biases = [b-(eta/len(batch))*nb for b,nb in zip(self.biases, b)]
            
    def backprop(self,x,y):
        update_w = [np.zeros(weight.shape) for weight in self.weights]
        update_b = [np.zeros(bias.shape) for bias in self.biases]
        #feedfoward
        activations = []
        activation = x
        activations.append(x)
        zs = []
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight) + bias
            zs.append(z)
            #print(zs)
            activation = sigmoid(z)
            #print(activation)
            activations.append(activation)
        #backpass
        delta = self.cost_derivative(activations[-1],y)*sig_prime(zs[-1])
        update_b[-1]  = delta
        update_w[-1] = np.dot(activations[-2].T, delta)
        #print(zs)
        for j in range(2,self.num_layers):
            sp = sig_prime(zs[-j])
            delta = np.dot(delta, self.weights[-j+1].T) * sp
            update_b[-j] = delta
            update_w[-j] = np.dot(activations[-j-1].T, delta)
        return(update_w, update_b)
    def evaluate(self, test_data):
        correct_count = 0
        for x, y in test_data:
            pred = np.argmax(self.feedfoward(x))
            if pred == np.argmax(y):
                correct_count += 1
            #correct_count += pred/len(test_data)
        return(correct_count)
    def cost_derivative(self,pred,y):
        return(pred-y)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sig_prime(z):
    return(sigmoid(z)*(1 - sigmoid(z)))

train = mnist_load('mnist_train.csv', 10000)
test = mnist_load('mnist_test.csv', 500)
net = Network((784,30,10))
net.SGD(train, 10, 30, 3.0, test)