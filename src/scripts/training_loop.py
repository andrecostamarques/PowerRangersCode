"""
Training loop execute the training for the Config object specified.
It recieves the Config object and instantiate the trainer.

It main focus in only to setup the configuration of the training and to instantiate and call the correct methods.
"""
# Importing section 

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
from DatasetsDict import DatasetDict
from TotalLoss import TotalLoss

from LeNet5 import LeNet5
import SelectionMask as sm


def main():
    """
    Main execution function for the training script.

    Initializes the dataset, defines the configuration dictionary for the
    experiment, and starts the training process using the StaticMaskTraining class.
    """
    # Instantiating the DatasetDict
    db = DatasetDict()

    ds_list, tf_train, tf_test = db.get("mnist")

    mnist_raw_config = { 
        # Parameters for training
        "model": LeNet5(), 
        "n_epochs": 400, 
        "batch_size": 128, 
        "mask_shape": (1, 28, 28), 
        "model_learning_rate": 0.001, 
        "mask_learning_rate": 0.005, 
        
        # Params for the lambda scheduler
        "lambda_init": 0.0005, 
        "lambda_factor": 1.5, 
        "lambda_patience": 5,
        "lambda_treshold": 0.2, 
        
        # Params for the class identification
        "training_id": "lenet_mnist_02",
        "optimizer_class": optim.AdamW,
        "model_loss_function": nn.NLLLoss, # Correct loss for the LeNet5()
        "mask_model": sm.SelectionMask, # Our static Mask
        "mask_loss_function": sm.mask_l1_loss, # Our static Mask loss
        
        # Params for the dataset
        "datasets": ds_list, 
        "transform_train": tf_train, 
        "transform_test": tf_test,
        "eval_split": 0.1, # 10% of the dataset for validation
        "seed": 42,

        # Testing sending to another folder
        #"root_dir_save": "./test/"
    }

    # Instantiating the config file and the trainers
    config = TrainingConfig(**mnist_raw_config)   
    trainer = StaticMaskTraining(config) 
    trainer.train()

if __name__ == "__main__":
    main()
