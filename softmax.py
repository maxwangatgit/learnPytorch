#encoding=utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l
import numpy as np


# realize the sortmax function
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# loss function
def crossEntropy(y_hat, y):
    # get the target class probality
    # assume only one class in one image 
    output = y_hat.gather(1, y.view(-1, 1))

    # return the crossentropy
    return -torch.log(output)

# calculate the accuracy of the prediction 
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# define the struct of softmax net
def softmaxNet(X, num_inputs, w, b):
    # mm means mulit function of mat
    return softmax(torch.mm(X.view((-1, num_inputs)), w) + b)




def easySoftmax():




    num_inputs = 784
    num_outputs = 10

    # create the weight mat with random params
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32, requires_grad=True)
    # create the bias mat
    b = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)

    # create the net
    lr = 0.3
    batch_size = 256
    net = softmaxNet
    loss = crossEntropy

    net = softmaxNet
    num_epochs = 5

   
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [w, b], lr)






easySoftmax()