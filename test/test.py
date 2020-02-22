import numpy as np
import math
import struct
import matplotlib.pyplot as plt
from pathlib import Path

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()

dimensions = [28*28, 10]
activation = [tanh, softmax]
distribution = [
    {'b': [0,0]},
    {'b': [0,0],'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])), math.sqrt(6/(dimensions[0]+dimensions[1]))]},
]

def init_parameters_b(layer):
    dist = distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameters_w(layer):
    dist = distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]

def init_parameters():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in range(i):
            if j == 'b':
                layer_parameter['b'] = init_parameters_b(i)
                continue
            if j == 'w':
                layer_parameter['w'] = init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter

parameters = init_parameters()

def predict(img, parameters):
    l0_in = img +parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out,parameters[1]['w']+parameters[1]['b'])
    l1_out = activation[1](l1_in)
    return l1_out

dataset_path = Path('./MNIST')
train_img_path = dataset_path/'train-images.idx3-ubyte'
train_lab_path = dataset_path/'train-labels.idx1-ubyte'
test_img_path = dataset_path/'t10k-images.idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels.idx1-ubyte'

train_num = 50000
valid_num = 10000
test_num = 10000


with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1,28*28)
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1,28*28)

with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]

with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)

def show_train(index):
    plt.imshow(train_img[index].reshape(28,28), cmap = 'gray')
    print('label:{}'.format(train_lab[index]))

def show_valid(index):
    plt.imshow(valid_img[index].reshape(28,28), cmap = 'gray')
    print('label:{}'.format(valid_lab[index]))

def show_test(index):
    plt.imshow(test_img[index].reshape(28,28), cmap = 'gray')
    print('label:{}'.format(test_lab[index]))


show_train(np.random.randint(train_num))
show_valid(np.random.randint(valid_num))
show_test(np.random.randint(test_num))