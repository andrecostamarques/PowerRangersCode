#!/usr/bin/env python3
import SelectionMask as sm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

"""
Training Configuration Module.

This module defines the `TrainingConfig` class, which serves as a centralized
configuration object for setting up training experiments. It holds hyperparameters,
model definitions, dataset settings, and scheduling parameters.
"""

class TrainingConfig:
    """
    Configuration holder for training experiments.

    This class encapsulates all necessary parameters to configure a training run,
    including model architecture, optimization settings, dataset transformations,
    and lambda scheduling for sparsity control. It also provides utility methods
    to generate dataloaders based on the stored configuration.

    Attributes:
        optimizer_class (type): The class of the optimizer to be used (e.g., torch.optim.Adam).
        transform_train (transforms.Compose): Transformations for the training dataset.
        transform_test (transforms.Compose): Transformations for the test/validation dataset.
        eval_split (float): Fraction of the training data to be used for validation.
        datasets (list): A list containing [train_dataset, test_dataset].
        seed (int): Random seed for reproducibility.
        model (nn.Module): The neural network model instance.
        model_loss_function (type): The class of the loss function for the model (e.g., nn.CrossEntropyLoss).
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for dataloaders.
        mask_shape (tuple): Shape of the selection mask.
        model_learning_rate (float): Learning rate for the main model parameters.
        mask_learning_rate (float): Learning rate for the mask parameters.
        lambda_init (float): Initial value for the sparsity regularization weight (lambda).
        lambda_factor (float): Multiplicative factor for updating lambda.
        lambda_patience (int): Number of epochs to wait before updating lambda.
        lambda_treshold (float): Threshold for relative improvement to reset patience.
        training_id (str): Identifier for the training run.
        mask_loss_function (callable): Function to compute the mask sparsity loss.
        mask_model (type): The class of the mask model to be instantiated.
        root_dir_save (str): Root directory for saving checkpoints.
    """
    
    def __init__(self, optimizer_class, transform_train, transform_test, eval_split, datasets, seed, model, 
                model_loss_function, n_epochs, batch_size, mask_shape, model_learning_rate, mask_learning_rate, lambda_init, 
                lambda_factor, lambda_patience, lambda_treshold, training_id,  
                mask_loss_function=sm.mask_l1_loss, mask_model = sm.SelectionMask, root_dir_save = '../../checkpoints/'):
        """
        Initializes the TrainingConfig object.

        Args:
            optimizer_class (type): The optimizer class (e.g., torch.optim.Adam).
            transform_train (torchvision.transforms.Compose): Training data transformations.
            transform_test (torchvision.transforms.Compose): Test/Validation data transformations.
            eval_split (float): Percentage of training data to split for validation (0.0 to 1.0).
            datasets (list): List containing the raw [train_dataset, test_dataset].
            seed (int): Random seed for data splitting and reproducibility.
            model (torch.nn.Module): The instantiated model architecture.
            model_loss_function (type): The loss function class (e.g., nn.NLLLoss).
            n_epochs (int): Total number of training epochs.
            batch_size (int): Size of data batches.
            mask_shape (tuple): Dimensions of the input mask (e.g., (1, 28, 28)).
            model_learning_rate (float): Learning rate for the classification model.
            mask_learning_rate (float): Learning rate for the mask.
            lambda_init (float): Initial lambda value for sparsity regularization.
            lambda_factor (float): Factor to increase lambda by when patience runs out.
            lambda_patience (int): Epochs to wait for improvement before increasing lambda.
            lambda_treshold (float): Minimum relative improvement required to reset patience.
            training_id (str): Unique ID for the experiment (used for folder naming).
            mask_loss_function (callable, optional): Function to calculate mask loss. Defaults to sm.mask_l1_loss.
            mask_model (type, optional): Class for the mask model. Defaults to sm.SelectionMask.
            root_dir_save (str, optional): Directory to save checkpoints. Defaults to '../../checkpoints/'.
        """
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
        """
        Prepares and returns DataLoaders for training, validation, and testing.

        Splits the training dataset into training and validation subsets based on
        `eval_split` and creates DataLoaders for all sets using the configured
        batch size and seed.

        Args:
            seed (int, optional): Random seed for the split generator. Defaults to 42.

        Returns:
            tuple: A tuple containing (train_loader, val_loader, test_loader).
        """
        
        generator = torch.Generator('cpu').manual_seed(seed)

        train_dataset = self.datasets[0]
        test_dataset = self.datasets[1]

        n_val = int(len(train_dataset) * self.eval_split)
        n_train = len(train_dataset) - n_val
        train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        dataset_name = self.datasets[0].__class__.__name__
        print(f"Dataloaders initialized for dataset: {dataset_name}")
        print(len(train_subset), len(val_subset), len(test_dataset))
        
        return train_loader, val_loader, test_loader
