'''
Author: your name
Date: 2020-10-27 17:10:22
LastEditTime: 2020-10-27 17:23:43
LastEditors: Please set LastEditors
Description: In User Settings Edit

FilePath: \learnPytorch\convel.py
'''
import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] -h + 1, X.shape[1] - w + 1 ))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j]= (X[i:i+h,j:j+w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

