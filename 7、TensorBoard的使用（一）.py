#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 19:39
# Author  : dongchao
# File    : 7、TensorBoard的使用（一）.py
# Software: PyCharm


from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter("logs")
    for i in range(0, 100):
        writer.add_scalar("y = 2*x", 2 * i, i)  # 不同的图使用不同的tag
    writer.close()
