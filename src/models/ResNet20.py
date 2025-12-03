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
    Standard residual block for ResNet architectures.
    
    Consists of two 3x3 convolutions with Batch Normalization and ReLU.
    Includes a shortcut connection that adapts dimensions using a 1x1 convolution
    if the stride is greater than 1 or channel dimensions change.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the first convolution. Defaults to 1.
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    """
    ResNet architecture implementation optimized for CIFAR-10/100 (32x32 input).
    
    This implementation differs from standard ImageNet ResNets by using a 
    smaller initial kernel (3x3 vs 7x7) and removing the initial max pooling 
    to preserve spatial dimensions for small images.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Args:
            block (nn.Module): The residual block class to use (e.g., BasicBlock).
            num_blocks (list[int]): A list of length 3 specifying the number of blocks per layer.
            num_classes (int): Number of classification classes. Defaults to 10.
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
        Creates a sequential stack of residual blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    """
    Constructs a ResNet-20 model for CIFAR.

    Configuration:
        - Depth: 20 layers (6n + 2, where n=3)
        - Structure: [3, 3, 3] blocks per stage.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ResNet_CIFAR: The initialized model.
    """
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)