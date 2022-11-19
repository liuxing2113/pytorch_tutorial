# -*- coding: utf-8 -*-
# @时间: 2022/11/17 2022/11/17
# @作者： 流星
# @项目：**


import torch
import torchvision
from PIL import Image
from model import *

image_path = "../imgs/airplane2.png"
image = Image.open(image_path)

image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
model = torch.load("../model/CIFAR10_model_9.pth")
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
target = output.argmax(1).item()
print(target)