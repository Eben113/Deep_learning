import sys
import numpy as np
from collections import Counter
import math
np.random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

f = open('labels.txt')
raw_labels = f.readlines()
f.close()

tokens = list(map(lambda x: set(x.split(" ")), raw_reviews))

vocab = set()
for sent in tokens:
	for j in sent:
		if len(j) >0:
			vocab.add(j)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
	word2index[word] = i

input_dset = []
for i in range(len(tokens)):
	sent_indices = list()
	for j in tokens[i]:
		try:
			sent_indices.append(word2index[j])
		except:
			" "
	input_dset.append(sent_indices)

targ_dset = list()
for label in raw_labels:
	if label == 'positive\n':
		targ_dset.append(1)
	else:
		targ_dset.append(0)

alpha, iterations = 0.01,2
hidden_size = 100
def sigmoid(x):
	return(1/(1+np.exp(-x)))

weight01 = 0.2*np.random.random((len(vocab),hidden_size))-0.1
weight12 = 0.2*np.random.random((hidden_size,1))-0.1
for iter in range(iterations):
	correct, total = 0,0
	for i in range(len(input_dset)-1000):
		x,y = input_dset[i], targ_dset[i]
		l1 = sigmoid(np.sum(weight01[x],axis=0))
		l2 = sigmoid(l1.dot(weight12))
		
		l2_delta = l2 - y
		l1_delta = l2_delta.dot(weight12.T)
		
		weight01[x] -= l1_delta*alpha
		weight12 -= np.outer(l1,l2_delta)*alpha
		
		if np.abs(l2_delta) < 0.5:
			correct += 1
		total += 1
		if i%10 == 9:
			progress = str(i/len(input_dset))
			print('Iter:{}---progress: {}.{}---Train Acc:{}---------'.format(iter,progress[2:4],progress[4:6],correct/float(total)))
for i in range(len(input_dset)-1000,len(input_dset)):
	correct,total = 0,0
	x,y = input_dset[i],targ_dset[i]
	T1 = sigmoid(np.sum(weight01[x],axis=0))
	T2 = sigmoid(T1.dot(weight12))
	
	if (T2-y) < 0.5:
		correct += 1
	total += 1
print('Test Acc: {}'.format(correct/float(total)))
def similar(target = 'beautiful'):
	target_index = word2index[target]
	score = Counter()
	for word,i in word2index.items():
		diff = weight01[i] - weight01[target_index]
		sq_diff = diff*diff
		score[word] =-math.sqrt(sum(sq_diff))
	return score.most_common(30)
print(similar('terrible'))