'''
Author: your name
Date: 2020-10-23 19:59:17
LastEditTime: 2020-10-23 21:13:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \learnPytorch\mlp.py
'''
import numpy as np
import numpy
import matplotlib.pylab as plt
import sys
import torch
sys.path.append("..") 
import d2lzh_pytorch as d2l


def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel("x")
    d2l.plt.ylabel(name+"(x)")
    d2l.plt.show()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, W2, b1, b2]

for param in params:
    params.requires_grad_(requires_grad=True)

#define the relu
def relu(X):
    return torch.max(input=x, other=torch.tensor(0.0))

# define the forward net
def net(X):
    X = X.view(-1, num_inputs)
    H = relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H, W2) + b2

loss = torch.nn.CrossEntropyLoss()

num_epochs,lr = 5, 100.0

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, None)
