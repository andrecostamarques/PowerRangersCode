"""
Simple Convolutional Neural Network Module.

This module implements a basic CNN architecture suitable for simple image classification
tasks like MNIST. It features three convolutional layers with max pooling, dropout for
regularization, and fully connected layers.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network architecture.

    This network consists of three convolutional layers with ReLU activation and max pooling,
    followed by dropout and two fully connected layers. It is designed for single-channel
    input images (e.g., 28x28 grayscale).

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (1 -> 32 channels).
        conv2 (nn.Conv2d): Second convolutional layer (32 -> 64 channels).
        conv3 (nn.Conv2d): Third convolutional layer (64 -> 128 channels).
        pool (nn.MaxPool2d): Max pooling layer used after each convolution.
        dropout1 (nn.Dropout): Dropout layer applied after the convolutional block.
        dropout2 (nn.Dropout): Dropout layer applied before the final output.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Output fully connected layer.
        softmax (nn.LogSoftmax): Output activation function.
    """
    def __init__(self):
        """
        Initializes the SimpleCNN model layers.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Log-probabilities for each class, shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x