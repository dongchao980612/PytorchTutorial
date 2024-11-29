#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 18:57
# Author  : dongchao
# File    : 1、PyTorch环境的配置及安装.py
# Software: PyCharm

import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())  # True
