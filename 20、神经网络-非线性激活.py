#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 12:04
# Author  : dongchao
# File    : 20、神经网络-非线性激活.py
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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.relu(input)
        return output


if __name__ == '__main__':
    x = torch.tensor([
        [1, -0.5],
        [-1, 3],
    ])

    x = torch.reshape(x, (-1, 1, 2, 2))
    net = Model()

    # print(net(x))

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

