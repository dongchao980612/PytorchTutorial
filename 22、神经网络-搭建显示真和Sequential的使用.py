#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 12:40
# Author  : dongchao
# File    : 22、神经网络-搭建显示真和Sequential的使用.py
# Software: PyCharm


import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, (5, 5), padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, (5, 5), padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, (5, 5), padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    dataset = CIFAR10("./datasets", train=False, transform=ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)
    x = torch.rand(64, 3, 32, 32)

    net = Model()
    # print(net(x).shape)

    writer = SummaryWriter('logs')
    writer.add_graph(net, x)
    writer.close()
