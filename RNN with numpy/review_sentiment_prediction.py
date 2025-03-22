import numpy as np
from collections import Counter
import math
np.random.seed(42)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

f = open('labels.txt')
raw_labels = f.readlines()
f.close()

vocab = set()
for i in raw_reviews:
    for j in i.split(' '):
        if len(j) > 0:
            vocab.add(j)
        
idx_lookup = {}
i = 0
for word in vocab:
    idx_lookup[word] = i
    i += 1

input_ = []
for review in raw_reviews:
    indices = []
    for word in review.split(' '):
        if len(word) > 0:
            indices.append(idx_lookup[word])
    input_.append(indices)
    
    


labels = np.zeros((len(raw_labels), 1))
for i in range(len(raw_labels)):
    if raw_labels[i] ==  'positive\n':
        labels[i] = 1
    else:
        labels[i] = 0

def sigmoid(x):
    return(1/(1+np.exp(-x)))
def softmax(x):
    temp = np.exp(x)
    return(temp)
def sigderiv(x):
    return(x*(1-x))
train_size = len(raw_labels) - 1000
hidden_size = 100
weight_01 = 0.2*(np.random.rand(len(vocab), hidden_size))-0.1
weight_12 =0.2 *( np.random.rand(hidden_size, 1)) - 0.1
alpha = 0.01
iterations = 2


for iters in range(iterations):
    total, correct = 0,0
    for i in range(train_size):
        x,y = input_[i], labels[i]
        layer_1 = sigmoid(np.sum(weight_01[x], axis = 0))
        layer_2 = sigmoid(np.dot(weight_12.T, layer_1))
        
        l2_delta = layer_2 - y
        l1_delta = weight_12.dot(l2_delta)*sigderiv(layer_1)
        
        weight_12 -= alpha*np.outer(layer_1, l2_delta)
        weight_01[x] -= alpha*l1_delta
        
        if np.abs(l2_delta) < 0.5:
            correct += 1
        
        total += 1
        progress = str(i/train_size)
        if i%10 == 9:
            print('Iter:{} | Progress:{}.{}% | Accuracy:{}'.format(iters, progress[2:4], progress[4:6], correct/float(total)))
            #print(weight_01[195])
    
    correct, total = 0,0
    for i in range(len(raw_labels)-train_size):
        x,y = input_[i], labels[i]
        layer_1 = sigmoid(np.sum(weight_01[x], axis = 0))
        layer_2 = sigmoid(layer_1.dot(weight_12))
        
        if np.abs(layer_2 - y) < 0.5:
            correct += 1
        total += 1
    print('Iter: {} | Accuracy: {}'.format(iters, correct/float(total)))
    
def similar(word):
    list_ = Counter()
    array = weight_01[idx_lookup[word]]
    for i in vocab:
        array_i = weight_01[idx_lookup[i]]
        list_[i] = -math.sqrt(((array_i-array)**2).mean())
    print(list_.most_common(10))
print(similar('terrible'))