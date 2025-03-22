import sys, random
import numpy as np
from collections import Counter
import math
np.random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x: x.split(' '), raw_reviews))
wordcnt = Counter() 
vocab = set()
for token in tokens:
	for word in token:
		wordcnt[word] -= 1
vocab = list(set(map(lambda x:x[0], wordcnt.most_common() )))

word2index = {}
for i,word in enumerate(vocab):
	word2index[word] = i

concatenate = []
input_dataset = []
for sent in tokens:
	sent_indices = []
	for word in sent:
		try:
			concatenate.append(word2index[word])
			sent_indices.append(word2index[word])
		except:
			' '
	input_dataset.append(sent_indices)

concatenated = np.array(concatenate)
random.shuffle(input_dataset)

alpha, iterations = 0.05, 2
hidden_size, window, negative = 50, 2, 5

weight01 = 0.2 * np.random.rand(len(vocab), hidden_size) -0.5
weight12 = np.random.rand(len(vocab), hidden_size)

layer2target = np.zeros(negative+1)
layer2target[0] = 1

def similar(word):
	targ_index = word2index[word]
	count = Counter()
	for word,index in word2index.items():
		raw_difference = weight01[index] - weight01[targ_index]
		sq_diff = raw_difference * raw_difference
		dist = -np.mean(math.sqrt(sq_diff))
		count[word] = dist
	return(count.most_common(10))

def sigmoid(x):
	return(1/(1+np.exp(-x)))

for i,review in enumerate(input_dataset * iterations):
	for targ in range(len(review)):
		targ_sample = [review[targ]] + list(concatenated[np.random.rand((negative)*len(concatenated)).astype('int').tolist()])
		left = review[max(0,targ-window):targ]
		right = review[targ+1:min(targ+window, len(review))]
		l1 = np.mean(weight01[left + right], axis=0)
		l2 = sigmoid(l1.dot(weight12[targ_sample].T))
		l2_delta = l2-layer2target
		l1_delta = l2_delta.dot(weight12[targ_sample])
		
		weight12[target_sample] -= np.outer(l2_delta, l1)*alpha
		weight01[left + right] -= l1_delta*alpha
		if i%250 == 0:
			print('progress: {}----\n{}'.format(i/len(input_dataset)*iterations,similar('terrible')))
		print('progress: {}----\n{}'.format(i/len(input_dataset)*iterations,similar('terrible')))
print(similar('terrible'))