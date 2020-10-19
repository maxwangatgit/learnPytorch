'''
Author: your name
Date: 2020-10-19 15:43:07
LastEditTime: 2020-10-19 18:42:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \learnPytorch\softmax.py
'''
#encoding=utf-8

import torch
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l
import numpy as np
from torch import nn
from torch.nn import init

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    # x shape:(batch, *, *)
    def forward(self,x):
        return x.view(x.shape[0],-1)




class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


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
def softmaxNet(X, w, b):
    # the net should init the params of it set
    # mm means mulit function of mat
    num_inputs = 784
    return softmax(torch.mm(X.view((-1, num_inputs)), w) + b)




# a complete achievement of softmax 
def completeSoftmax():
    num_inputs = 784
    num_outputs = 10

    # create the weight mat with random params
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32, requires_grad=True)
    # create the bias mat
    b = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)

    # create the net
    lr = 0.2
    batch_size = 32
    net = softmaxNet
    loss = crossEntropy

    net = softmaxNet
    num_epochs = 100

   
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [w, b], lr)




# a esay achievement of softmax by using nn
def easySoftmax():
    num_inputs = 784
    num_outputs = 10
    batch_size = 32
    # create the forward net
    net = LinearNet(num_inputs, num_outputs)

    net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear' , nn.Linear(num_inputs, num_outputs))
        ])
    )

    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 50
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)




easySoftmax()






