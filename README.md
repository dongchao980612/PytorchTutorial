#  PyTorch教程代码-土堆
- 作者：[我是土堆](https://space.bilibili.com/203989554)

- 代码作者：[dongchao](https://github.com/dongchao980612)

- 讲解视频：[https://www.bilibili.com/video/BV1hE411t7RN/](https://www.bilibili.com/video/BV1hE411t7RN/)


## P5-6、PyTorch加载数据初认识
数据集：[蚂蚁蜜蜂分类数据集](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

```python
from torch.utils.data import Dataset

class MyData(Dataset):

     def __init__(self):
         pass

     def __getitem__(self, item):
         pass
```

## p7-8、TensorBoard的使用
- add_scalar
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
for i in range(0, 100):
    writer.add_scalar("y = x", i, i)
writer.close()
# tensorboard  --logdir=logs
```
- add_image【**PIL==9.5.0**】

```python
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import PIL

print(PIL.__version__)  # 9.5.0
img_path = "datasets/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
print(type(img))
image_array = np.array(img)
print(type(image_array), image_array.shape)  # (512, 768, 3)

writer = SummaryWriter("logs")
writer.add_image('numpy img', image_array, 1, dataformats="HWC")
writer.close()
```