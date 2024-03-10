# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : TestTorch.py
# Time       ：2023/9/6 22:39
# Author     ：Cheng Jungao
# version    ：python 3.9
# Description：
"""

import torch
import numpy as np

print(torch.cuda.is_available())

a = torch.randn(2, 3, 3)
print(a)

print(a.dim())  # 查看维度
print(a.shape)  # 查看形状

b = torch.tensor(1.0)

print(type(b))

c = np.ones(2)
print(c)
d = torch.tensor(c)
print(d)
