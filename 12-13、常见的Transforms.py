#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 20:19
# Author  : dongchao
# File    : 12-13、常见的Transforms.py
# Software: PyCharm

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    img_path = "datasets/hymenoptera_data/train/ants/0013035.jpg"
    img = Image.open(img_path)

    writer = SummaryWriter('logs')

    trans_ToTensor = transforms.ToTensor()
    img_Totensor = trans_ToTensor(img)
    writer.add_image('image tensor', img_Totensor, 1)

    trans_Normalize = transforms.Normalize([5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_Totensor_Normalize = trans_Normalize(img_Totensor)
    writer.add_image('image tensor normalize', img_Totensor_Normalize, 1)

    trans_Resize = transforms.Resize((256, 256))
    img = trans_Resize(img)
    img_Totensor_Normalize_Resize=trans_ToTensor(img)
    writer.add_image('image tensor normalize resize', img_Totensor_Normalize_Resize, 1)

    trans_Resize_2 = transforms.Resize(256)
    trans_Compose = transforms.Compose([
        trans_Resize_2,
        trans_ToTensor
    ])
    img_Totensor_Normalize_Resize_Compose= trans_Compose(img)
    writer.add_image('image tensor normalize resize compose', img_Totensor_Normalize_Resize_Compose, 1)

    trans_RandomCrop = transforms.RandomCrop(256)
    trans_Compose_2 = transforms.Compose([
        trans_RandomCrop,
        trans_ToTensor
    ])
    for i in range(5):
        img_Totensor_Normalize_Resize_Compose_2 = trans_Compose_2(img)
        writer.add_image('image tensor normalize resize compose2', img_Totensor_Normalize_Resize_Compose_2, i)

    writer.close()
