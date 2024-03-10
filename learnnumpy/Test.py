# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Test.py
# Time       ：2023/10/18 22:57
# Author     ：Cheng Jungao
# version    ：python 3.9
# Description：
"""
import numpy as np

# 创建一个numpy数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 找到整个数组的最大值
print(np.max(arr))  # 输出9

# 找到数组中每个元素的最大值
print(np.max(arr, axis=0))  # 输出[7 8 9]，即每列的最大值
print(np.max(arr, axis=1))  # 输出[3 6 9]，即每行的最大值
