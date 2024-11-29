#  PyTorch教程代码-土堆
- 作者：[我是土堆](https://space.bilibili.com/203989554)

- 代码作者：[dongchao](https://github.com/dongchao980612)

- 讲解视频：[https://www.bilibili.com/video/BV1hE411t7RN/](https://www.bilibili.com/video/BV1hE411t7RN/)


## P5
数据集：[蚂蚁蜜蜂分类数据集](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

```python
from torch.utils.data import Dataset

class MyData(Dataset):

     def __init__(self):
         pass

     def __getitem__(self, item):
         pass
```