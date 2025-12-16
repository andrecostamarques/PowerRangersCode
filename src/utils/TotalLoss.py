import torch
import torch.nn as nn
from LambdaScheduler import *

class TotalLoss(nn.Module):
    def __init__(self, model_loss, mask_loss_function, lambda_scheduler : LambdaScheduler):
        super(TotalLoss, self).__init__()
        self.model_loss = model_loss
        self.lambda_scheduler = lambda_scheduler
        self.mask_loss_function = mask_loss_function

    def forward(self, pred, target, mask_model):
        model_loss = self.model_loss(pred, target)
        mask_loss = self.mask_loss_function(mask_model)
        total = model_loss + self.lambda_scheduler.lbd * mask_loss
        return model_loss, mask_loss, total