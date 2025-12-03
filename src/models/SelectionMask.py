#!/usr/bin/env python3
"""
Selection Mask Module.

This module implements a learnable binary mask using the Straight-Through Estimator (STE).
It allows for the optimization of hard selection (binary) parameters within a differentiable
pipeline by approximating gradients during the backward pass.

Classes:
    SelectionMask: A neural network module that learns a broadcastable binary mask.

Functions:
    mask_l1_loss: Computes the sparsity regularization loss for the mask.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class SelectionMask(nn.Module):    
    """
    A module that learns a binary mask using the Straight-Through Estimator (STE).

    This allows the network to learn a hard selection (0 or 1) while maintaining 
    gradient flow during backpropagation.
    """
    def __init__(self, shape, mean=2, std=0.01) -> None:
        """
        Initializes the mask parameters.

        Args:
            shape (tuple): The dimensions of the learnable mask.
            mean (float, optional): Mean for the normal initialization. Defaults to 2.
            std (float, optional): Standard deviation for the normal initialization. Defaults to 0.01.
        """
        super().__init__()

        tensor = torch.randn(shape, requires_grad=True)
        self.mask = nn.Parameter(nn.init.normal_(tensor=tensor, mean=mean, std=std))
        
    def forward(self, x) -> torch.Tensor:
        """
        Applies the binarized mask to the input tensor.

        Uses the Straight-Through Estimator logic: y = hard_threshold(x) for the forward pass,
        but dy/dx = 1 (approximated via sigmoid) for the backward pass.
        If the mask shape differs from the input, it interpolates the mask to match.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor multiplied by the binary mask.
        """
        sig = torch.sigmoid(self.mask)
        bin_mask = torch.round(sig).float()
        diff_mask = bin_mask + (sig - sig.detach())

        if x.shape != self.mask.shape:
            diff_mask_scaled = F.interpolate(diff_mask.unsqueeze(0), size=x.shape[2:], mode='nearest')
            diff_mask = diff_mask_scaled.squeeze(0).expand_as(x)
            
        return x * diff_mask


def mask_l1_loss(mask: SelectionMask):
    """
    Computes the sparsity ratio (percentage of active elements) of the mask.

    Args:
        mask (SelectionMask): The mask module instance.

    Returns:
        torch.Tensor: Scalar value representing the fraction of the mask with value 1.
    """
    sig = torch.sigmoid(mask.mask)
    bin_mask = torch.round(sig).int()
    diff_mask = bin_mask + (sig - sig.detach())
    return diff_mask.sum() / diff_mask.numel()