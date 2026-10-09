#!/usr/bin/env python3
"""
Experiment execution script.
Runs the training loop for any dataset (Galaxy10, Food-101, CIFAR-10) across all four models:
- ResNet20 (3 channels, 256x256)
- LeNet256 (3 channels, 256x256)
- SimpleCNNRGB (3 channels, 256x256)
- ResNet34 (3 channels, 256x256)

Each model has custom configuration, including optimized lambdas and loss functions.
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import torch
import torch.nn as nn
import torch.optim as optim

from StaticMaskTraining import StaticMaskTraining
from TrainingConfig import TrainingConfig
from DatasetsDict import DatasetDict

from LeNet256 import LeNet256
from ResNet20 import resnet20
from SimpleCNNRGB import SimpleCNNRGB
from torchvision.models import resnet34
import SelectionMask as sm


def get_num_classes(dataset_name):
    if dataset_name.lower() == 'food101':
        return 101
    return 10


def run_model(model_name, dataset_name="galaxy10", epochs=200, resume=False):
    print("\n" + "="*60)
    print(f"STARTING EXPERIMENT: {model_name.upper()} on {dataset_name.upper()}")
    print("="*60)
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get(dataset_name)
    num_classes = get_num_classes(dataset_name)
    
    if model_name == 'resnet20':
        model = resnet20(num_classes=num_classes)
        loss_fn = nn.CrossEntropyLoss
        lr = 1e-4
        lambda_init = 1.0
        lambda_patience = 2
    elif model_name == 'resnet34':
        model = resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        loss_fn = nn.CrossEntropyLoss
        lr = 1e-4
        lambda_init = 1.0
        lambda_patience = 2
    elif model_name == 'lenet256':
        model = LeNet256(num_classes=num_classes)
        loss_fn = nn.NLLLoss
        lr = 1e-3
        lambda_init = 0.1
        lambda_patience = 3
    elif model_name == 'simplecnnrgb':
        model = SimpleCNNRGB(num_classes=num_classes)
        loss_fn = nn.NLLLoss
        lr = 1e-3
        lambda_init = 0.1
        lambda_patience = 3
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    training_id = f"{dataset_name.lower()}_{model_name}_200epochs"

    config_dict = { 
        "model": model, 
        "n_epochs": epochs, 
        "batch_size": 64, 
        "mask_shape": (3, 256, 256), 
        "model_learning_rate": lr, 
        "mask_learning_rate": 0.001, 
        
        "lambda_init": lambda_init, 
        "lambda_factor": 1.5, 
        "lambda_patience": lambda_patience,
        "lambda_treshold": 0.0025, 
        
        "training_id": training_id,
        "optimizer_class": optim.AdamW,
        "model_loss_function": loss_fn,
        "mask_model": sm.SelectionMask,
        "mask_loss_function": sm.mask_l1_loss,
        
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }
    
    config = TrainingConfig(**config_dict)   
    trainer = StaticMaskTraining(config) 
    trainer.train(resume=resume)


def main():
    parser = argparse.ArgumentParser(description="Run training experiments across multiple models and datasets.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["all", "resnet20", "resnet34", "lenet256", "simplecnnrgb"], 
        default="all",
        help="Specify which model to train. Default is 'all'."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="galaxy10",
        help="Specify dataset from DatasetDict (e.g. galaxy10, cifar10, food101). Default is galaxy10."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train. Default is 200."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume training from the latest valid checkpoint."
    )
    args = parser.parse_args()

    models_to_run = ["lenet256", "resnet20", "simplecnnrgb", "resnet34"] if args.model == "all" else [args.model]
    
    for m in models_to_run:
        run_model(m, dataset_name=args.dataset, epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()
