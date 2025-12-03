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
    LeNet-5 Convolutional Neural Network.

    A classic architecture composed of two convolutional feature extraction blocks
    followed by fully connected classification layers.

    Architecture:
        - Input: 1 channel (grayscale).
        - Layer 1: Conv2d (6 filters) + AvgPool + Tanh.
        - Layer 2: Conv2d (16 filters) + AvgPool + Tanh.
        - Layer 3: Conv2d (120 filters) + Tanh.
        - Classifier: Fully connected layers (120*3*3 -> 84 -> 10).
    """
    def __init__(self):
        """
        Initializes the LeNet-5 layers.
        """
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classification = nn.Sequential(
            nn.Linear(120 * 3 * 3, 84),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28) or similar.

        Returns:
            torch.Tensor: Log-softmax output probabilities.
        """
        x = self.features(x)
        x = self.classification(x)

        return x