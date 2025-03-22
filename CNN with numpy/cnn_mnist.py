import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset
train = datasets.load_from_disk('/storage/emulated/0/python/train')
test = datasets.load_from_disk('/storage/emulated/0/python/test')
a = []
b = []
for i in range(1000):
	a.append(np.array(train[i]['image']).reshape(1,784))
a = (np.concatenate(a))/255
for i in range(100):
	b.append(np.array(np.array(test[i]['image']).reshape(1,784)))
b = (np.concatenate(b))/255
labels = np.array(train[0:1000]['label'])
test_labels = np.array(test[0:100]['label'])
label = np.zeros((1000,10))
test_label= np.zeros((100,10))
for i,l in enumerate(labels.flat):
	label[i][l] = 1
for i,l in enumerate(test_labels.flat):
	test_label[i][l] = 1
np.random.seed(1)
def tanh(x):
	return np.tanh(x)
def tanh2deriv(output):
	return(1 - (output)**2)
def softmax(x):
	temp = np.exp(x)
	return(temp/np.sum(temp, axis = 1, keepdims = True))
relu = lambda x: (x>=0)*x
relu2deriv = lambda x: (x>=0)
alpha = 2
batch_size = 128
in_rows = 28
in_cols = 28
kernel_rows = 3
kernel_cols = 3
no_kernels = 16
hidden_size = ((in_rows - kernel_rows)*(in_cols-kernel_cols)* no_kernels)
kernels =0.02* (np.random.random((kernel_rows*kernel_cols,no_kernels))) -0.01
weight12 = 0.2*(np.random.random((hidden_size,10))) -0.1
def get_img_sect(layer,r_from,r_to,c_from,c_to):
	sub_sect = layer[:,r_from:r_to,c_from:c_to]
	return sub_sect.reshape(-1,1,r_to-r_from,c_to-c_from)
for i in range(350):
	error,correct_cnt = 0.0,0
	for j in range(int(len(a)/batch_size)):
		batch_start, batch_end = (j*batch_size),((j+1)*batch_size)
		l0 = a[batch_start:batch_end]
		l0 = l0.reshape(l0.shape[0],28,28)
		l0.shape
		
		sects = list()
		for r_start in range(in_rows-kernel_rows):
			for col_start in range(in_cols - kernel_cols):
				sect = get_img_sect(l0,r_start,r_start+kernel_rows,col_start,col_start+kernel_cols)
				sects.append(sect)
		expanded_in = np.concatenate(sects,axis= 1)
		es = expanded_in.shape
		flat_in = expanded_in.reshape(es[0]*es[1],-1)
		kernel_out = np.dot(flat_in,kernels)
		l1 = tanh(kernel_out.reshape(es[0],-1))
		dropout_mask = np.random.randint(2,size = l1.shape)
		l1 *= dropout_mask*2
		l2 = softmax(np.dot(l1,weight12))
		for k in range(batch_size):
			_inc = int(np.argmax(l2[k:k+1]) == np.argmax(label[batch_start+k:batch_start+k+1]))
			correct_cnt += _inc
		delta12 = (label[batch_start:batch_end] - l2)/(batch_size*l2.shape[0])
		delta01 = (delta12.dot(weight12.T))*tanh2deriv(l1)
		delta01 *= dropout_mask
		weight12 += alpha*(l1.T.dot(delta12))
		l1d_reshape = delta01.reshape(kernel_out.shape)
		k_update = flat_in.T.dot(l1d_reshape)
		kernels += alpha*k_update
		
		test_correct_cnt = 0
	for j in range(len(b)):
		    l0 = b[j:j+1]
		    l0 = l0.reshape(l0.shape[0],28,28)
		    l0.shape
		    sects = list()
		    for r_start in range(in_rows-kernel_rows):
		    	for col_start in range(in_cols - kernel_cols):
		    		sect = get_img_sect(l0,r_start,r_start+kernel_rows,col_start,col_start+kernel_cols)
		    		sects.append(sect)
		    expanded_in = np.concatenate(sects,axis= 1)
		    es = expanded_in.shape
		    flat_in = expanded_in.reshape(es[0]*es[1],-1)
		    kernel_out = np.dot(flat_in,kernels)
		    l1 = tanh(kernel_out.reshape(es[0],-1))
		    l2 = np.dot(l1,weight12)
		    test_correct_cnt += int(np.argmax(l2)== np.argmax(test_label[j:j+1]))
	if i%10 == 0 or i == 349:
			print('i: {}'.format(i))
			print('	Correct: {} '.format(correct_cnt/float(len(a))))
			print('	test_Correct: {}'.format(str(test_correct_cnt/len(b))))