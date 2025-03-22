import numpy as np
images = [point[0] for point in train]
labels = [point[1] for point in train]
test_ = [point[0] for point in test]
test_labels = [point[1] for point in test]
def tanh(x):
    return np.tanh(x)
def tanh_delta(x):
    return 1 - tanh(x)**2
def softmax(x):
    temp = np.sum(np.exp(x))
    return np.exp(x)/temp
image_size, labels, batch_size, = 784, 10, 128
alpha, iterations = 2, 300
input_rows, input_cols = 28,28
kernel_rows, kernel_cols, num_kernels = 3,3,16
num_labels = 10
hidden_size = ((input_rows - kernel_rows)* (input_cols-kernel_cols)* num_kernels)
kernels = 0.02*np.random.rand((kernel_rows*kernel_cols), num_kernels)-0.01
w12 = 0.2*np.random.rand(hidden_size, num_labels)-0.1
def get_sections(layer, rf, rt, cf, ct):
    sect = layer[:, rf:rt, cf:ct]
    return sect.reshape(-1,1,rt-rf, ct-cf)
for j in range(iterations):
    correct_cnt = 0
    for i in range(len(images)/ batch_size):
        layer_0 = images[i*batch_size:(i+1)*batch_size]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        
        sects = []
        for row_index in range(input_rows - kernel_rows):
            for col_index in range(input_cols-kernel_cols):
                sects.append(get_sections(layer_0, row_index, row_index+kernel_rows, col_index,
                                          col_index+kernel_cols))
        expanded =  np.concatenate(sects, axis=1)
        es = expanded.shape
        flatten = expanded.reshape(es[0]*es[1], -1)
        
        kernel_out = flatten.dot(kernels)
        layer_1 = tanh(kernel_out.reshape(es[0], -1))
        dropout = np.random.randint(2, layer_0.shape)
        layer_1 *= dropout
        layer_2 = softmax(np.dot(layer_1, w12))
        
        for k in range(batch_size):
            label = labels[i*batch_size+k: i*batch_size+k+1]
            correct_cnt += int(np.argmax(label) == np.argmax(layer_2[k:k+1]))
        
        l2_delta  = (labels[i*batch_size+k: i*batch_size+k+1] - layer_2)/(batch_size * layer_2.shape[0])
        l1_delta = l2_delta.dot(w12.T)*tanh_delta(layer_1)
        w12 +=  alpha*layer_1.T.dot(l2_delta)
        l1_delta *= dropout
        l1_delta = l1_delta.reshape(kernel_out)
        kernels -= alpha*flatten.T.dot(l1_delta)
    test_cnt = 0 
    for i in range(len(test_)):
        l0 = test_[i:i+1]
        l0 = l0.reshape(l0.shape[0], 28, 28)
        
        sects = []
        for row_index in range(input_rows - kernel_rows):
            for col_index in range(input_cols-kernel_cols):
                sects.append(get_sections(l0, row_index, row_index+kernel_rows, col_index,
                                          col_index+kernel_cols))
        expanded = np.concatenate(sects, axis = 0)
        es = expanded.shape
        flatten = expanded.reshape(es[0]*es[1], -1)
        
        kernel_out = flatten.dot(kernels)
        l1 = tanh(kernel_out.reshape(es[0], -1))
        l2 = softmax(l1.dot(w12))
        
        test_cnt += int(np.argmax(l2) == np.argmax(test_labels[i:i+1]))
    print('I:{} Test-Acc:{} Train-Acc:{}'.format(j, (100*correct_cnt)/len(images), 
                                                 (100*test_cnt)/len(test_)))