#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 21:19
# Author  : dongchao
# File    : 15、DataLoader的使用.py
# Software: PyCharm


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    transforms = transforms.Compose([
        ToTensor()
    ])

    CIFAR10_train_set = CIFAR10(root="./datasets", train=True, transform=transforms, download=False)
    CIFAR10_train_loader = DataLoader(dataset=CIFAR10_train_set, batch_size=16, shuffle=True, num_workers=0,
                                      drop_last=False)

    # 使用board可视化
    writer = SummaryWriter("logs")

    epochs = 2
    for e in range(epochs):
        step = 0
        for data in CIFAR10_train_loader:
            image, targets = data
            # print(image.shape,targets) # torch.Size([4, 3, 32, 32]) tensor([3, 0, 2, 9])
            writer.add_images("Epoch:{}".format(e), image, step)
            step = step + 1
    writer.close()
