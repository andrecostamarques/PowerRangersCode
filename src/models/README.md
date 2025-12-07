# Models

This directory contains the neural network model definitions and the custom mask layer.

## Available Modules

### `SelectionMask`
A learnable binary mask module implemented using the Straight-Through Estimator (STE). It allows the network to learn which features to keep (multiplication by 1) or discard (multiplication by 0) in an end-to-end differentiable manner.

### `LeNet5`
Implementation of the classic LeNet-5 architecture.

### `ResNet20`
Implementation of ResNet-20, suitable for CIFAR-10/100 tasks.
