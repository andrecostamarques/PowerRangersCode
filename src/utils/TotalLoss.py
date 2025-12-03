import torch
import torch.nn as nn
from LambdaScheduler import *

class TotalLoss(nn.Module):
    def __init__(self, model_loss, lambda_scheduler : LambdaScheduler):
        super(TotalLoss, self).__init__()
        self.model_loss = model_loss
        self.lambda_scheduler = lambda_scheduler

    def forward(self, pred, target, mask_loss):
        model_loss = self.model_loss(pred, target)
        total = model_loss + self.lambda_scheduler.lbd * mask_loss
        return model_loss, total