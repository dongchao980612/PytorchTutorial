#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 12:29
# Author  : dongchao
# File    : 21、线形层和其他层.py
# Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


if __name__ == '__main__':
    dataset = CIFAR10("./datasets", train=False, transform=ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=64,drop_last=True)

    net = Model()

    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)
        output = torch.flatten(imgs)
        print(output.shape)
        output = net(output)
        print(output.shape)
