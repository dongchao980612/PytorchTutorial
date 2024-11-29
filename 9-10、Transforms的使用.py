#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 20:07
# Author  : dongchao
# File    : 9-10、Transforms的使用.py
# Software: PyCharm


from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    img_path = "datasets/hymenoptera_data/train/ants/0013035.jpg"
    img = Image.open(img_path)

    writer = SummaryWriter('logs')

    # print(type(img))  # PIL
    trans_ToTensor = transforms.ToTensor()
    # img_tensor = trans_ToTensor.__call__(img)
    img_Totensor = trans_ToTensor(img)
    # print(type(img_Totensor))  # Tensor

    writer.add_image('tensor image', img_Totensor, 1)

    writer.close()


