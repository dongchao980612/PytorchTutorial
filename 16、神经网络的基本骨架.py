#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/30 11:06
# Author  : dongchao
# File    : 16、神经网络的基本骨架.py
# Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input = input + 1
        return input


if __name__ == '__main__':
    net = Model()
    x = torch.tensor(1)  # torch.int64
    output = net(x)
    print(output)

    x = torch.tensor([
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ])
    weight = torch.tensor([
        [1, 2, 1],
        [0, 1, 0],
        [2, 1, 0]
    ])
    x = torch.reshape(x, (1, 1, 5, 5))
    weight = torch.reshape(weight, (1, 1, 3, 3))

    print(x.shape)
    print(weight.shape)

    output1 = F.conv2d(x, weight, stride=1)
    print(output1, output1.shape)

    output2 = F.conv2d(x, weight, stride=2)
    print(output2, output2.shape)
