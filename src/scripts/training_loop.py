import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

from StaticMaskTraining import StaticMaskTraining

from TrainingConfig import TrainingConfig
from LambdaScheduler import LambdaScheduler
from TotalLoss import TotalLoss

from LeNet5 import LeNet5
import SelectionMask as sm


def main():
    mnist_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    mnist_datasets = [
        datasets.MNIST(root='../../data/', train=True, download=True, transform=mnist_transform),
        datasets.MNIST(root='../../data/', train=False, download=True, transform=mnist_transform)
    ]

    mnist_raw_config = { 
        # Parâmetros de Modelo/Treinamento
        "model": LeNet5(), 
        "n_epochs": 400, 
        "batch_size": 128, 
        "mask_shape": (1, 28, 28), 
        "model_learning_rate": 0.001, 
        "mask_learning_rate": 0.005, 
        
        # Parâmetros do Lambda Scheduler
        "lambda_init": 0.0005, 
        "lambda_factor": 1.5, 
        "lambda_patience": 5,
        "lambda_treshold": 0.2, 
        
        # Parâmetros de Identificação e Classes
        "training_id": "lenet_mnist_01",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.NLLLoss, # Loss compatível com LogSoftmax
        "mask_model": sm.SelectionMask, # Sua classe de máscara
        "mask_loss_function": sm.mask_l1_loss, # Sua função de perda de esparsidade
        
        # Parâmetros de Dados (Exigidos pela sua TrainingConfig)
        "datasets": mnist_datasets, 
        "transform_train": mnist_transform, 
        "transform_test": mnist_transform,
        "eval_split": 0.1, # 10% do dataset de treino será usado para validação
        "seed": 42,

        # Testando enviar em outra pasta
        #"root_dir_save": "./teste/"
    }

    config = TrainingConfig(**mnist_raw_config)
    trainer = StaticMaskTraining(config)
    trainer.train()

if __name__ == "__main__":
    main()

