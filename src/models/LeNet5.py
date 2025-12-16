#!/usr/bin/env python3
"""
LeNet-5 Architecture Implementation.

This module contains the implementation of the classic LeNet-5 Convolutional Neural Network
as originally proposed by Yann LeCun et al. for handwritten digit recognition (MNIST).
It utilizes average pooling and Tanh activations, characteristic of early CNN designs.

Classes:
    LeNet5: The main network architecture.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120 * 3 * 3, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1) 
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x