#!/usr/bin/env python3
"""
ResNet Architecture for CIFAR-10/100.

This module implements the ResNet architecture specifically adapted for small input sizes
(32x32), as described in "Deep Residual Learning for Image Recognition" (He et al.).
It removes the initial 7x7 convolution and max pooling found in ImageNet variants to
preserve spatial resolution in deeper layers.

Classes:
    BasicBlock: Standard residual block with two 3x3 convolutions.
    ResNet_CIFAR: Main network definition for variable depth.

Functions:
    resnet20: Factory function to create a ResNet-20 model.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Standard Residual Block for ResNet.

    Consists of two 3x3 convolutions with Batch Normalization and ReLU activation.
    Includes a shortcut connection (identity mapping or projection) to handle
    dimension matching.

    Attributes:
        conv1 (nn.Conv2d): First 3x3 convolution.
        bn1 (nn.BatchNorm2d): Batch norm after first conv.
        conv2 (nn.Conv2d): Second 3x3 convolution.
        bn2 (nn.BatchNorm2d): Batch norm after second conv.
        shortcut (nn.Sequential): Shortcut connection (identity or 1x1 conv).
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        """
        Initializes the BasicBlock.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the first convolution. Defaults to 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual connection and activation.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    """
    ResNet architecture adapted for CIFAR-10/100 data.

    This implementation uses 3 groups of residual layers with increasing channel counts
    (16, 32, 64). It is designed for 32x32 input images.

    Attributes:
        in_planes (int): Tracker for current number of input channels during layer construction.
        conv1 (nn.Conv2d): Initial 3x3 convolution.
        bn1 (nn.BatchNorm2d): Initial batch normalization.
        layer1 (nn.Sequential): First group of residual blocks (16 channels).
        layer2 (nn.Sequential): Second group of residual blocks (32 channels).
        layer3 (nn.Sequential): Third group of residual blocks (64 channels).
        linear (nn.Linear): Final fully connected layer for classification.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Initializes the ResNet model.

        Args:
            block (class): The residual block class to use (e.g., BasicBlock).
            num_blocks (list of int): Number of blocks in each of the 3 layers.
            num_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a sequence of residual blocks.

        Args:
            block (class): The block class.
            planes (int): Number of output channels for this layer.
            num_blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block in the sequence.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        """
        Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    """
    Constructs a ResNet-20 model.

    ResNet-20 consists of 3 groups of 3 blocks each (plus initial conv and final fc),
    totaling 6*3 + 2 = 20 layers.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        ResNet_CIFAR: The configured ResNet-20 model.
    """
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)