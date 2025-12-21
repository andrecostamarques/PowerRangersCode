from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda")

NUM_CLIENTS = 10
BATCH_SIZE = 32


import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def load_datasets(partition_id: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download do dataset completo
    full_trainset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Divisão em N partições (clientes)
    partition_size = len(full_trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    # Ajusta sobra se a divisão não for exata
    lengths[-1] += len(full_trainset) - sum(lengths)
    
    datasets_per_client = random_split(full_trainset, lengths, generator=torch.Generator().manual_seed(42))

    # Seleciona a partição do cliente atual
    client_dataset = datasets_per_client[partition_id]

    # Divide a partição do cliente: 80% treino, 20% validação
    len_val = int(len(client_dataset) * 0.2)
    len_train = len(client_dataset) - len_val
    train_ds, val_ds = random_split(client_dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader

'''======================= CLIENT ==========================='''
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    # Returns current local parameters
    def get_parameters(self, config):
        
        #returns type "NDArrays" list
        pass
    
    # Receives upper parameters, trains with local data, returns new parameters
    def fit(self, parameters, config):
        
        #returns type "tuple[NDArrays, int, dict[str, Scalar]]"
        pass

    # Receives upper parameters, evalu
    def evaluate(self, parameters, config):
        
        #returns type "tuple[float, int, dict[str, Scalar]]"
        pass

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    
    # net = Net().to(DEVICE)

    # partition_id = context.node_config["partition-id"]
    # trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # return FlowerClient(net, trainloader, valloader).to_client()
    
    pass