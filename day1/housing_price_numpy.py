# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : housing_price_numpy.py
# Time       ：2023/10/23 22:25
# Author     ：Cheng Jungao
# version    ：python 3.9
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data(datafile):
    # 从文件导入数据
    data = np.fromfile(datafile, sep=' ')

    """
    14 features 分别为：犯罪率、超过25平方英尺住宅用地比例、非零售商业用地比例、是否靠河、空气质量、房间数、
    年限、距离、高速公路、税率、学生/教师比例、黑人比例、低地位人口比例、房价
    """
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)

    # 对数据进行归一化处理，将值处理为0-1之间的数
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network1(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        """
        前向传播,计算预测值y
        :param x:
        :return:
        """
        z = np.dot(x, self.w) + self.b  # 线性回归,z = w * x + b
        return z

    def loss(self, z, y):
        """
        计算损失函数值,均方误差 MSE
        :param z:
        :param y:
        :return:
        """
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples  # 均方误差
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x  # 损失函数对w求导,结果是一个矩阵，形状与x相同
        gradient_w = np.mean(gradient_w, axis=0)  # 求每一列的平均值
        gradient_w = gradient_w[:, np.newaxis]  # 将行向量转换为列向量
        gradient_b = (z - y)  # 损失函数对b求导,结果是一个矩阵，形状与y相同
        gradient_b = np.mean(gradient_b)  # 求平均值
        return gradient_w, gradient_b  # 返回梯度

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)  # 保存每次迭代的损失函数值
            if (i + 1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses


if __name__ == '__main__':
    # 获取数据
    train_data, test_data = load_data('../data/housing.data')
    x = train_data[:, :-1]
    y = train_data[:, -1:]

    # 创建网络
    net = Network1(13)
    num_iterations = 50
    # 启动训练
    losses = net.train(x, y, iterations=num_iterations, eta=0.01)

    print("w:", net.w)
    print("b:", net.b)

    # 预测
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1:]
    losses_test = net.loss(net.forward(x_test), y_test)
    print("测试集损失函数值：", losses_test)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()
