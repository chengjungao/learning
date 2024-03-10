# _*_ coding : utf-8 _*_ 
# coding=utf-8
# @Time : 2024/3/10 19:09 
# @Author : chengjungao 
# @File : housing_price_torch 
# @Project : learning
import numpy
import torch
from torch import nn
import torch.optim as optim


def load_data(datafile):
    # 从文件导入数据
    data = numpy.fromfile(datafile, sep=' ')

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
    return torch.tensor(training_data, dtype=torch.float32), torch.tensor(test_data, dtype=torch.float32)


# 定义一个线性回归网络
net = nn.Sequential(nn.Linear(13, 1))

# 打印网络结构
print(net)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 开始训练
num_epochs = 100
train_data, test_data = load_data('../data/housing.data')
X = train_data[:, :-1]
y = train_data[:, -1:]
for epoch in range(1, num_epochs + 1):
    # 前向传播
    y_pred = net(X)
    l = loss(y_pred, y)
    # 反向传播
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


# 保存模型
torch.save(net, 'model.pkl')

# 使用模型预测
X_test = test_data[:, :-1]
y_test = test_data[:, -1:]
y_pred_test = net(X_test)
test_loss = loss(y_pred_test, y_test)
print('test loss: %f' % test_loss.item())

