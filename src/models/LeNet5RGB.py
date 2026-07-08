#!/usr/bin/env python3
"""
LeNet-5 RGB Architecture Implementation.
Adapted for 3-channel (RGB) images of 256x256 resolution.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class LeNet5RGB(nn.Module):
    """
    LeNet-5 Convolutional Neural Network architecture adapted for RGB 256x256 images.
    Uses AdaptiveAvgPool2d to reduce spatial dimensions to 3x3 before fully connected layers.
    """
    def __init__(self):
        super(LeNet5RGB, self).__init__()
        # 3 input channels instead of 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        
        # Adaptive pooling to ensure 3x3 feature map before linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        self.fc1 = nn.Linear(120 * 3 * 3, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        """
        Defines the forward pass.
        Expects shape (batch_size, 3, 256, 256).
        """
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        
        # Resize to 3x3 before flattening
        x = self.adaptive_pool(x)
        
        x = torch.flatten(x, 1) 
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
