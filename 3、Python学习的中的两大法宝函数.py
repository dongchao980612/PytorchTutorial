#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 19:09
# Author  : dongchao
# File    : 2、Python学习的中的两大法宝函数.py
# Software: PyCharm
import  torch
if __name__ == '__main__':

    print(dir(torch.cuda.is_available))
    print(help(torch.cuda.is_available))