#!/usr/bin/env python3
"""
Simple Convolutional Neural Network RGB Module.
Adapted for 3-channel (RGB) images of 256x256 resolution.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class SimpleCNNRGB(nn.Module):
    """
    A simple Convolutional Neural Network architecture adapted for RGB 256x256 images.
    """
    def __init__(self):
        super(SimpleCNNRGB, self).__init__()
        # 3 input channels instead of 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Input to fc1 is 128 * 32 * 32 for 256x256 input
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Defines the forward pass.
        Expects shape (batch_size, 3, 256, 256).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
