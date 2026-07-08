#!/usr/bin/env python3
"""
Experiment execution script.
Runs the training loop for Galaxy10 dataset across all three models:
- ResNet20 (3 channels, 256x256)
- LeNet5RGB (3 channels, 256x256)
- SimpleCNNRGB (3 channels, 256x256)

Each model has custom configuration, including optimized lambdas and loss functions.
All models run on the native RGB 256x256 Galaxy10 dataset.
"""

import sys
import os
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
import SelectionMask as sm


def run_resnet20(resume=False):
    print("\n" + "="*50)
    print("STARTING EXPERIMENT: ResNet20 on Galaxy10")
    print("="*50)
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get("galaxy10")
    
    resnet_config = { 
        "model": resnet20(), 
        "n_epochs": 200, 
        "batch_size": 64, 
        "mask_shape": (3, 256, 256), 
        "model_learning_rate": 1e-4, 
        "mask_learning_rate": 0.001, 
        
        # Params for the lambda scheduler
        "lambda_init": 1.0, 
        "lambda_factor": 1.5, 
        "lambda_patience": 2,
        "lambda_treshold": 0.0025, 
        
        # Params for class/training identification
        "training_id": "galaxy10_resnet20_200epochs",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.CrossEntropyLoss, # ResNet20 outputs logits
        "mask_model": sm.SelectionMask,
        "mask_loss_function": sm.mask_l1_loss,
        
        # Params for the dataset
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }
    
    config = TrainingConfig(**resnet_config)   
    trainer = StaticMaskTraining(config) 
    trainer.train(resume=resume)


def run_lenet256(resume=False):
    print("\n" + "="*50)
    print("STARTING EXPERIMENT: LeNet256 on Galaxy10")
    print("="*50)
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get("galaxy10")
    
    lenet_config = { 
        "model": LeNet256(), 
        "n_epochs": 200, 
        "batch_size": 64, 
        "mask_shape": (3, 256, 256), 
        "model_learning_rate": 1e-3, # Standard for LeNet256 with AdamW
        "mask_learning_rate": 0.001, 
        
        # Params for the lambda scheduler
        # LeNet256 has lower capacity, start with a smaller lambda to let it learn first
        "lambda_init": 0.1, 
        "lambda_factor": 1.5, 
        "lambda_patience": 3,
        "lambda_treshold": 0.0025, 
        
        # Params for class/training identification
        "training_id": "galaxy10_lenet256_200epochs",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.NLLLoss, # LeNet256 outputs log-softmax
        "mask_model": sm.SelectionMask,
        "mask_loss_function": sm.mask_l1_loss,
        
        # Params for the dataset
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }
    
    config = TrainingConfig(**lenet_config)   
    trainer = StaticMaskTraining(config) 
    trainer.train(resume=resume)


def run_simplecnnrgb(resume=False):
    print("\n" + "="*50)
    print("STARTING EXPERIMENT: SimpleCNNRGB on Galaxy10")
    print("="*50)
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get("galaxy10")
    
    simplecnn_config = { 
        "model": SimpleCNNRGB(), 
        "n_epochs": 200, 
        "batch_size": 64, 
        "mask_shape": (3, 256, 256), 
        "model_learning_rate": 1e-3, 
        "mask_learning_rate": 0.001, 
        
        # Params for the lambda scheduler
        # SimpleCNNRGB has lower capacity than ResNet, start with a smaller lambda
        "lambda_init": 0.1, 
        "lambda_factor": 1.5, 
        "lambda_patience": 3,
        "lambda_treshold": 0.0025, 
        
        # Params for class/training identification
        "training_id": "galaxy10_simplecnnrgb_200epochs",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.NLLLoss, # SimpleCNNRGB outputs log-softmax
        "mask_model": sm.SelectionMask,
        "mask_loss_function": sm.mask_l1_loss,
        
        # Params for the dataset
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }
    
    config = TrainingConfig(**simplecnn_config)   
    trainer = StaticMaskTraining(config) 
    trainer.train(resume=resume)


def run_resnet34(resume=False):
    print("\n" + "="*50)
    print("STARTING EXPERIMENT: ResNet34 on Galaxy10")
    print("="*50)
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get("galaxy10")
    
    from torchvision.models import resnet34
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    resnet_config = { 
        "model": model, 
        "n_epochs": 300, 
        "batch_size": 64, 
        "mask_shape": (3, 256, 256), 
        "model_learning_rate": 1e-4, 
        "mask_learning_rate": 0.001, 
        
        # Params for the lambda scheduler
        "lambda_init": 1.0, 
        "lambda_factor": 1.5, 
        "lambda_patience": 2,
        "lambda_treshold": 0.0025, 
        
        # Params for class/training identification
        "training_id": "galaxy10_resnet34",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.CrossEntropyLoss,
        "mask_model": sm.SelectionMask,
        "mask_loss_function": sm.mask_l1_loss,
        
        # Params for the dataset
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }
    
    config = TrainingConfig(**resnet_config)   
    trainer = StaticMaskTraining(config) 
    trainer.train(resume=resume)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run training experiments for Galaxy10 across multiple models.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["all", "resnet20", "resnet34", "lenet256", "simplecnnrgb"], 
        default="all",
        help="Specify which model to train. Default is 'all'."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume training from the latest valid checkpoint."
    )
    args = parser.parse_args()

    if args.model == "all":
        run_resnet20(resume=args.resume)
        run_resnet34(resume=args.resume)
        run_lenet256(resume=args.resume)
        run_simplecnnrgb(resume=args.resume)
    elif args.model == "resnet20":
        run_resnet20(resume=args.resume)
    elif args.model == "resnet34":
        run_resnet34(resume=args.resume)
    elif args.model == "lenet256":
        run_lenet256(resume=args.resume)
    elif args.model == "simplecnnrgb":
        run_simplecnnrgb(resume=args.resume)


if __name__ == "__main__":
    main()
