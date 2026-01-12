import torch
import torch.nn.functional as F
import torch.nn as nn

class LeNet5_256(nn.Module):
    def __init__(self):
        super(LeNet5_256, self).__init__()
        # 1. Mudança para 3 canais de entrada (RGB)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        
        # 2. O SEGREDO: Garante que o mapa de features seja 3x3 antes da FC
        # independente se a imagem começou com 28 ou 256 pixels.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        self.fc1 = nn.Linear(120 * 3 * 3, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        
        # Redimensiona para 3x3 antes de achatar
        x = self.adaptive_pool(x)
        
        x = torch.flatten(x, 1) 
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x