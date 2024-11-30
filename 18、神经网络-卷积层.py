#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 11:25
# Author  : dongchao
# File    : 18、神经网络-卷积层.py
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    dataset = CIFAR10("./datasets", train=False, transform=ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=64)

    net = Model()
    print(net)

    writer = SummaryWriter('logs')

    step = 0
    for data in dataloader:
        imgs, target = data
        output = net(imgs)
        # print(imgs.shape, output.shape)  # torch.Size([64, 3, 32, 32]) torch.Size([64, 6, 30, 30])

        writer.add_images("input image", imgs, step)
        output = torch.reshape(output, (-1, 3, 30, 30))
        # print(output.shape)  # torch.Size([128, 3, 30, 30
        writer.add_images("output image", output, step)
        step = step + 1

    writer.close()
