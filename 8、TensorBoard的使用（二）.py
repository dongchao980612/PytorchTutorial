#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2024/11/29 19:51
# Author  : dongchao
# File    : 8、TensorBoard的使用（二）.py
# Software: PyCharm


import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import PIL

if __name__ == '__main__':
    # PIL -> ndarray
    print(PIL.__version__)  # 9.5.0
    img_path = "datasets/hymenoptera_data/train/ants/0013035.jpg"
    img = Image.open(img_path)
    print(type(img))
    image_array = np.array(img)
    print(type(image_array), image_array.shape)  # (512, 768, 3)

    writer = SummaryWriter("logs")
    writer.add_image('numpy img', image_array, 1, dataformats="HWC")
    writer.close()
