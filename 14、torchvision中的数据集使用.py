#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 21:05
# Author  : dongchao
# File    : 14、torchvision中的数据集使用.py
# Software: PyCharm


from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    transforms = transforms.Compose([
        ToTensor()
    ])

    CIFAR10_train_set = CIFAR10(root="./datasets", train=True, transform=transforms, download=False)
    CIFAR10_test_set = CIFAR10(root="./datasets", train=False, transform=transforms, download=False)

    # print(CIFAR10_train_set[0])
    print(CIFAR10_train_set.classes)
    print(CIFAR10_train_set.class_to_idx)

    image, target = CIFAR10_train_set[0]
    # image.show()
    print(target, CIFAR10_train_set.classes[target])

    writer = SummaryWriter('logs')
    for i in range(10):
        img, target = CIFAR10_train_set[i]
        writer.add_image('CIFAR10 dataset', img, i)

    writer.close()
