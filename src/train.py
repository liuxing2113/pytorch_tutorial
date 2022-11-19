# -*- coding: utf-8 -*-
# @时间: 2022/11/17 2022/11/17
# @作者： 流星
# @项目：**


# 准备数据集
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=True)


# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{}\n测试数据集长度：{}".format(train_data_size, test_data_size))


# 利用dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络
CIFAR10_test = Cifar10()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# 定义学习率
# learning_rate = 0.01
learning_rate = 1e-2
optimize = torch.optim.SGD(CIFAR10_test.parameters(), lr = learning_rate)

# 设置训练网络的一些参数
# 纪律训练的册书
total_train_step = 0

# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("="*20, "第{}轮训练开始".format(i+1), "="*20)
    # 训练开始
    CIFAR10_test.train()
    for data in train_dataloader:
        imgs, targets = data
        output = CIFAR10_test(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        # 记录训练次数
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("*"*20, "训练次数: {}, Loss: {}".format(total_train_step, loss.item()), "*"*20)
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    CIFAR10_test.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = CIFAR10_test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("-"*20, "整体测试集上的Loss: {}".format(total_test_loss), "-"*20)
    print("～"*20, "整体测试集上的正确率: {}".format(total_accuracy/test_data_size), "～"*20)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(CIFAR10_test, "../model/CIFAR10_model_{}.pth".format(i))
    print("模型已保存")

writer.close()