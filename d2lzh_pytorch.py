#!encoding=utf-8
import torch
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

import torchvision
import numpy as np
import random
import time
import sys

# reshape the x 


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# 本函数已保存在d2lzh包中方便以后使用
# 用于读取训练样本数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):  
    # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 输出线性回归的值
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): 
    # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size()))** 2 / 2


def sgd(params, lr, batch_size): 
     # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

# 本函数已保存在d2lzh包中方便以后使用
# get the text labels 
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size, resize):


    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    # download the train and test image data
    mnist_train = torchvision.datasets.FashionMNIST(root='d:/pytorch/Datasets/FashionMNIST', train=True, download=True, transform=transform)
    mnist_test  = torchvision.datasets.FashionMNIST(root='d:/pytorch/Datasets/FashionMNIST', train=False, download=True, transform=transform)


    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# calaulate the accurancy of data sets
def evaluate_accurancy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定device 就使用net的device
        device = list(net.parameters())[0].device


    acc_sum, n = 0.0, 0

    with torch.no_grad():            
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式，关掉dropout
                y_hat = net(X.to(device))
                y_hat = net(X.to(device)).argmax(dim=1)
                print(y_hat,"    ",y.to(device))
                print((y_hat == y.to(device)).float().sum().item())
                acc_sum += (y_hat == y.to(device)).float().sum().item()
                net.train() # 改回训练模式
            # there is y.shape[0] classes in one X
            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                     # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# 本函数已保存在d2lzh_pytorch包中方便以后使用
# achieve the tensor shape transform
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# optimizer : the param 
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):

    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    
    # train epoch times and epoch has batch_size images
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        
        for X, y in  train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
       
            # clean the grad of the net 
            n += y.shape[0]
            batch_count += 1


        test_acc = evaluate_accurancy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))




        
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

    plt.show()

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] -h + 1, X.shape[1] - w + 1 ))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j]= (X[i:i+h,j:j+w] * K).sum()
    return Y

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])