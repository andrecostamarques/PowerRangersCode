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
    """
    LeNet-5 Convolutional Neural Network architecture.

    This implementation follows the classic architecture designed for MNIST digit recognition.
    It consists of two convolutional layers with average pooling, followed by fully connected layers.
    Tanh activation functions are used throughout the network, except for the final softmax output.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool1 (nn.AvgPool2d): First average pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool2 (nn.AvgPool2d): Second average pooling layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Output fully connected layer.
        softmax (nn.LogSoftmax): Output activation.
    """
    def __init__(self):
        """
        Initializes the LeNet-5 model layers.
        """
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
        """
        Defines the forward pass of the LeNet-5 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Log-probabilities for each class, shape (batch_size, 10).
        """
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1) 
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x