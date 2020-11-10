
import torch
from torch import nn
import d2lzh_pytorch as d2l



class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

def comp_conv2d(conv2d, X):
    X = X.view((1, 1)+X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


def corr2d_multi_in_out(X, K):
    # 对K的第零维进行遍历，每次同输出X做互相计算。 所有的结果使用stack函数合并
    return torch.stack([corr2d_multi_in(X, k) for k in K])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            # kernel_size, stride
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
              nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output



net = LeNet()


batch_size = 4
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)