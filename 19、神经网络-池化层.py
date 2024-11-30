#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 11:44
# Author  : dongchao
# File    : 19、神经网络-池化层.py
# Software: PyCharm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


if __name__ == '__main__':
    x = torch.tensor([
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ])

    x = torch.reshape(x, (1, 1, 5, 5))  # torch.int64

    net = Model()
    # print(net(x), net(x).shape)

    dataset = CIFAR10("./datasets", train=False, transform=ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=64)

    writer = SummaryWriter('logs')
    step = 0
    for data in dataloader:
        imgs, targets = data
        writer.add_images("input", imgs, step)
        output = net(imgs)
        writer.add_images("output", output, step)
        step = step + 1
    writer.close()
