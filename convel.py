
import torch
from torch import nn
import d2lzh_pytorch as d2l
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def runLeNet():

    net = LeNet()
    batch_size = 4
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn)


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())

    return blk






def funNiN():
    net = nn.Sequential(nin_block(1, 96, kernel_size=11, stride=4, padding=0),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Dropout(0.5),

                        # output class is 10
                        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
                        GlobalAvgPool2d(),
                        d2l.FlattenLayer())

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)



class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # line 1 
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)

        #line 2
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        #line 3
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # line 4 
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))

        # 合并四个输出结果的通道
        return torch.cat((p1, p2, p3, p4), dim=1)
        

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (64, 128), (16, 32), 32 ),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   d2l.GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5, 
                    d2l.FlattenLayer(), nn.Linear(1024, 10))

net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))
X = torch.rand(1, 1, 96, 96)
for blk in net.children(): 
    X = blk(X)
    print('output shape: ', X.shape)
