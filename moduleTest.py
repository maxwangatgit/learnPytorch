'''
Author: your name
Date: 2020-10-27 15:32:59
LastEditTime: 2020-10-27 16:47:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \learnPytorch\moduleTest.py
'''
import numpy as np
import torch.nn as nn
import numpy
import matplotlib.pylab as plt
import sys
import torch
sys.path.append("..") 
import d2lzh_pytorch as d2l


X = torch.rand(2, 784)

class MLP(nn.Module):
    # define two layer with param, full connection layer
    def __init__(self, **kwargs):
        # user mlp parent
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act    = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # define the forward net
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential,self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        # the input is some modules
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    # define the net forward
    def forward(self, input):
        # self._modules will return a OrderedDict in the order of of add
        for module in self._modules.values():
            input  = module(input)
        return input


class CenterLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenterLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        return x - x.mean()

layer = CenterLayer()
print(layer(torch.tensor([1,12,3,4,5], dtype=torch.float)))

