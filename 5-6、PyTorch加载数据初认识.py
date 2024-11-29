#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 19:20
# Author  : dongchao
# File    : 5-6、PyTorch加载数据初认识.py
# Software: PyCharm
import os

from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        self.path = os.path.join(self.root_dir, self.label_dir)

        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    # 再写一个获取数据集长度的魔法函数
    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    # 获取蚂蚁数据集dataset
    root_dir = "datasets/hymenoptera_data/train"
    label_dir = "ants"
    ants_dataset = MyData(root_dir, label_dir)
    print(ants_dataset.__len__())  # 124
    image, label = ants_dataset[0]
    # image.show()

    # demo5：再来获取蜜蜂的数据集
    root_dir = "datasets/hymenoptera_data/train"
    label_dir = "bees"
    bees_dataset = MyData(root_dir, label_dir)
    print(bees_dataset.__len__())  # 121


    # dataset数据集拼接
    train_dataset = ants_dataset + bees_dataset
    print(train_dataset.__len__())# 245
