#!/usr/bin/env python3
import SelectionMask as sm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

"""
Class Big Description

Classes:
    TrainingConfig:
"""

class TrainingConfig:
    """
    Class Small Description

    """

    """
    Arguments: 
    transform_train: The series of transformations for applying on the training dataset,
    transform_test: The series of transformations for applying on the test and validation dataset,
    eval_split: The % of the training dataset set for evaluating,
    datasets: The datasest settings,
    model: The object of the model we are training,
    model_loss: The loss function of the model,
    n_epochs: The number of epochs to train the model,
    batch_size: The size of the batch,
    mask_shape: Shape of the mask applied,
    model_learning_rate: The learning rate for the classification model,
    mask_learning_rate: The learning rate for the mask model,
    lambda_init: Initial value for the lambda,
    lambda_factor: The factor in which the lambda is multiplied,
    lambda_patiance: The amount of epochs the lambda will wait before changing,
    lambda_treshold: The treshold in which we consider the lambda has changed,
    training_id: The ID of the training,
    mask_loss: The mask model loss function,
    """
    
    def __init__(self, optimizer_class, transform_train, transform_test, eval_split, datasets, seed, model, 
                model_loss_function, n_epochs, batch_size, mask_shape, model_learning_rate, mask_learning_rate, lambda_init, 
                lambda_factor, lambda_patience, lambda_treshold, training_id,  
                mask_loss_function=sm.mask_l1_loss, mask_model = sm.SelectionMask, root_dir_save = '../../checkpoints/'):
        # gerar o dataloader de evaluation
        # retornar um objeto que tenha as info importantes
        self.optimizer_class = optimizer_class
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.eval_split = eval_split
        self.datasets = datasets
        self.seed = seed
        self.model = model
        self.model_loss_function = model_loss_function
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mask_shape = mask_shape
        self.model_learning_rate = model_learning_rate
        self.mask_learning_rate = mask_learning_rate
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        self.lambda_patience = lambda_patience
        self.lambda_treshold = lambda_treshold
        self.training_id = training_id
        self.mask_loss_function = mask_loss_function
        self.mask_model = mask_model
        self.root_dir_save = root_dir_save
     
        

    def get_dataloader(self, seed=42):

        """Encapsulates all data preparation logic."""
        
        generator = torch.Generator('cpu').manual_seed(seed)

        train_dataset = self.datasets[0]
        test_dataset = self.datasets[1]

        n_val = int(len(train_dataset) * self.eval_split)
        n_train = len(train_dataset) - n_val
        train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(len(train_subset), len(val_subset), len(test_dataset))
        
        return train_loader, val_loader, test_loader

