# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : housing_price.py
# Time       ：2023/10/18 22:09
# Author     ：Cheng Jungao
# version    ：python 3.9
# Description：
"""
import numpy as np

data_file = "../data/housing.data"


"""
第一步：读取数据，并处理数据
"""
data = np.fromfile(data_file, sep=" ")
"""
14 features 分别为：犯罪率、超过25平方英尺住宅用地比例、非零售商业用地比例、是否靠河、空气质量、房间数、
年限、距离、高速公路、税率、学生/教师比例、黑人比例、低地位人口比例、房价
"""
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])

"""
第二步：数据集划分
80%的数据用于训练，20%的数据用于测试
"""
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]

# 进行数据归一化处理
maximums, minimums, avgs = \
    training_data.max(axis=0), \
    training_data.min(axis=0), \
    training_data.sum(axis=0) / training_data.shape[0]

for i in range(feature_num):
    # print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

# x 是训练数据的特征值，y 是训练数据的标签值
x = training_data[:, :-1]
y = training_data[:, -1:]

print(x.shape)
print(y.shape)
# 设计一个线性回归模型 y = w * x + b
np.random.seed(0)
w = np.random.random((feature_num-1, 1))
print(w)
b = 0

y0 = np.dot(x[0], w) + b

loss0 = np.abs(y0 - y[0])
print(loss0)


