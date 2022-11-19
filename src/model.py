# -*- coding: utf-8 -*-
# @时间: 2022/11/17 2022/11/17
# @作者： 流星
# @项目：**
import torch
from torch import nn


# 搭建CIFAR10训练网络
class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    CIFAR10_test = Cifar10()
    input = torch.ones((64, 3, 32, 32))
    output = CIFAR10_test(input)
    print(output.shape)