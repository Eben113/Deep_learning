import csv
import itertools
import numpy as np
def mnist_load(file, size):
    file = open(file)
    file = csv.reader(file)
    random = itertools.combinations(file, size+1)
    random = next(random)
    output = []
    for i in range(1,size+1):
        data = np.array(random[i][1:], dtype = 'int64').reshape(1,-1)/255
        label = vectorize(random[i][0])
        output.append((data, label))
    return(output)
def vectorize(x):
    out = np.zeros((1,10))
    out[0,int(x)] = 1
    return out