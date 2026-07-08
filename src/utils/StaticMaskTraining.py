"""
Static Mask Training Module.

This module defines the `StaticMaskTraining` class, which encapsulates the logic
for training a neural network with a learnable static selection mask. It handles
the training loop, validation, checkpointing, and logging of metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

from TrainingConfig import TrainingConfig
from LambdaScheduler import LambdaScheduler
from TotalLoss import TotalLoss

class StaticMaskTraining:
    """
    Orchestrator for training models with learnable static masks.

    This class manages the entire training lifecycle, including initialization of
    models and optimizers, execution of training and validation loops, loss calculation
    involving sparsity constraints (via lambda scheduling), and persistence of results.

    Attributes:
        config (TrainingConfig): Configuration object containing hyperparameters and settings.
        device (torch.device): The computation device (CPU or CUDA).
        seed (int): Random seed for reproducibility.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        model (nn.Module): The classification neural network.
        mask_model (nn.Module): The learnable mask module.
        criterion (nn.Module): The loss function for the classification task.
        lambda_scheduler (LambdaScheduler): Scheduler for the sparsity regularization weight.
        total_loss_calculator (TotalLoss): Module to compute the combined loss.
        optimizer (torch.optim.Optimizer): Optimizer for both model and mask parameters.
        training_id (str): Unique identifier for the training run.
        checkpoint_dir (str): Directory where checkpoints and logs are saved.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initializes the StaticMaskTraining instance.

        Sets up the device, dataloaders, models, loss functions, optimizer, and
        directory structures based on the provided configuration.

        Args:
            config (TrainingConfig): The configuration object containing all necessary
                parameters for training.
        """

        #Initializing the config file and the cuda device
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.seed

        
        #Initializing the dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.config.get_dataloader(self.seed)


        #Initializing the models, Mask and Classifier
        self.model = self.config.model.to(self.device)
        self.mask_model = self.config.mask_model(shape=self.config.mask_shape).to(self.device)


        #Initializing the TotalLoss and Scheduler
        self.criterion = self.config.model_loss_function()

        self.lambda_scheduler = LambdaScheduler(init = self.config.lambda_init, 
                                                factor = self.config.lambda_factor, 
                                                patience = self.config.lambda_patience, 
                                                treshold = self.config.lambda_treshold)
        
        self.total_loss_calculator = TotalLoss(
            model_loss= self.criterion,
            mask_loss_function= self.config.mask_loss_function,
            lambda_scheduler=self.lambda_scheduler
        )

        #Initializing the Optimizer
        self.optimizer_class = self.config.optimizer_class
        self.optimizer_kwargs = {}

        if self.optimizer_class == optim.SGD:
            self.optimizer_kwargs['momentum'] = 0.9
        if self.optimizer_class == optim.Adam or self.optimizer_class == optim.AdamW:
            self.optimizer_kwargs['amsgrad'] = True

        self.optimizer = self.optimizer_class(
            [
            {'params': self.model.parameters(), 'lr': self.config.model_learning_rate},
            {'params': self.mask_model.parameters(), 'lr': self.config.mask_learning_rate}
        ],
        **self.optimizer_kwargs
    )

        #Initilizing the Checkpoints 
        self.training_id = self.config.training_id

        root_dir = self.config.root_dir_save
        self.checkpoint_dir = os.path.abspath(os.path.join(root_dir, self.training_id))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0
    
        
    def train_epoch(self):
        """
        Executes a single training epoch.

        Iterates through the training dataloader, performs forward and backward passes,
        updates model and mask parameters, and calculates running losses.

        Returns:
            tuple: A tuple containing three floats:
                - avg_model_loss: Average classification loss for the epoch.
                - avg_mask_loss: Average sparsity/mask loss for the epoch.
                - avg_total_loss: Average combined loss for the epoch.
        """
        loader = self.train_loader
        #It sets the loader

        running_model_loss = 0.0
        running_mask_loss = 0.0
        running_total_loss = 0.0

        for X, y in loader: 
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            X_masked = self.mask_model(X)
            y_pred = self.model(X_masked)


            # It uses the already instantiated Loss Calculator to get the final loss
            model_loss, mask_loss, total_loss = self.total_loss_calculator(
                pred=y_pred,
                target=y,
                mask_model=self.mask_model    
            )           

            total_loss.backward()
            self.optimizer.step()

            running_model_loss += model_loss.item()
            running_mask_loss += mask_loss.item()
            running_total_loss += total_loss.item()
        
        avg_model_loss = running_model_loss / len(loader)
        avg_mask_loss = running_mask_loss / len(loader)
        avg_total_loss = running_total_loss / len(loader)

        return avg_model_loss, avg_mask_loss, avg_total_loss

    @torch.no_grad()
    def validate_epoch(self):
        """
        Evaluates the model on the validation dataset.

        Performs inference without gradient updates to calculate the confusion matrix
        and accuracy.

        Returns:
            tuple: A tuple containing:
                - cm (numpy.ndarray): Confusion matrix of predictions vs targets.
                - accuracy (float): Validation accuracy percentage.
        """
        #Gets the validation dataset

        loader = self.val_loader
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                X_masked = self.mask_model(X)
                y_pred = self.model(X_masked)

                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        #For logging reasons, we decided to return the accuracy as well, but the CM already have all the 
        #data we needed.
        cm = confusion_matrix(all_targets, all_predictions)
        accuracy = 100 * cm.diagonal().sum() / cm.sum()

        return cm, accuracy

    def save_checkpoint(self, epoch, cm, avg_total_loss, avg_model_loss, avg_mask_loss, accuracy):
        """
        Saves the training state to a checkpoint file.

        Args:
            epoch (int): The current epoch index.
            cm (numpy.ndarray): The confusion matrix for the current epoch.
            avg_total_loss (float): Average total loss for the epoch.
            avg_model_loss (float): Average model loss for the epoch.
            avg_mask_loss (float): Average mask loss for the epoch.
            accuracy (float): Validation accuracy for the epoch.
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'mask_state_dict': self.mask_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_loss': avg_total_loss,
            'model_loss': avg_model_loss,
            'mask_loss': avg_mask_loss,
            'accuracy': accuracy,
            'cm': cm,
            'lambda': self.lambda_scheduler.lbd
        }

        if epoch == 0:
        # Add the model and mask object to the first epoch to later be able to test the model more easily
            checkpoint['model_obj'] = self.model
            checkpoint['mask_model_obj'] = self.mask_model
            print(f">>> Epoch 1: Saving the Mask Model and Model for future testing.")

        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt')

    def log_training(self, epoch, avg_model_loss, avg_mask_loss, avg_total_loss, accuracy):
        """
        Logs training metrics to a CSV file.

        Appends the metrics for the current epoch to 'training_log.csv' in the
        checkpoint directory. Creates the file with a header if it does not exist.

        Args:
            epoch (int): The current epoch index.
            avg_model_loss (float): Average model loss.
            avg_mask_loss (float): Average mask loss.
            avg_total_loss (float): Average total loss.
            accuracy (float): Validation accuracy.
        """
        log_file = os.path.join(self.checkpoint_dir, 'training_log.csv')
    
        data = [
            epoch+1,
            avg_total_loss,
            avg_model_loss,
            avg_mask_loss,
            accuracy, # Already calculated on the validation
            self.lambda_scheduler.lbd,
            self.lambda_scheduler.count,
        ]
        
        header = [
            'epoch', 
            'total_loss', 
            'model_loss', 
            'mask_loss', 
            'val_accuracy', 
            'lambda_value',
            'lambda_patience_count',
        ]
        file_exists = os.path.exists(log_file)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(header)
            
            writer.writerow(data)

    def train(self, resume=False):
        """
        Executes the full training pipeline.

        Iterates for the number of epochs specified in the configuration. In each loop:
        1. Runs a training epoch.
        2. Updates the lambda parameter via the scheduler.
        3. Runs a validation epoch.
        4. Logs metrics to console and CSV.
        5. Saves a checkpoint.
        """
        if resume:
            import glob
            import re
            ckpt_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pt'))
            if ckpt_files:
                epochs = []
                for f in ckpt_files:
                    match = re.search(r'checkpoint_epoch_(\d+)\.pt', f)
                    if match:
                        epochs.append((int(match.group(1)), f))
                epochs.sort()
                
                loaded = False
                for epoch_num, ckpt_file in reversed(epochs):
                    try:
                        print(f"Attempting to load checkpoint: {ckpt_file}...")
                        checkpoint = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                        
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.mask_model.load_state_dict(checkpoint['mask_state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.lambda_scheduler.lbd = checkpoint['lambda']
                        
                        self.start_epoch = checkpoint['epoch'] # this is epoch + 1 of the checkpoint
                        print(f"Successfully loaded checkpoint from epoch {epoch_num}. Resuming training from epoch {self.start_epoch + 1}...")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load checkpoint {ckpt_file}: {e}. Trying previous one...")
                if not loaded:
                    print("No valid checkpoints could be loaded. Starting from scratch.")

        #Debbugin
        print(f"Starting training ({self.config.n_epochs} epochs) in {self.device}.")
        print(f"Checkpoints will be saved in: {self.checkpoint_dir}")
        
        for epoch in range(self.start_epoch, self.config.n_epochs):
            print(f"\nEpoch: {epoch+1}/{self.config.n_epochs}")

            #Training
            self.model.train()
            self.mask_model.train()

            avg_model_loss, avg_mask_loss, avg_total_loss = self.train_epoch()

            self.lambda_scheduler.adapt_lambda(avg_total_loss)

            #Evaluating
            self.model.eval()
            self.mask_model.eval()
            
            cm_array, accuracy = self.validate_epoch()

            #Log
            print(f"\nTotal Loss: {avg_total_loss:.4f}, Model Loss: {avg_model_loss:.4f}, Mask Loss: {avg_mask_loss:.4f}"
                f"\nLambda: {self.lambda_scheduler.lbd:.4f}, Lambda Patience Count: {self.lambda_scheduler.count}"
                f"\nAccuracy: {accuracy:.4f}") 
            
            self.log_training(epoch, avg_model_loss, avg_mask_loss, avg_total_loss, accuracy)

            self.save_checkpoint(epoch, cm_array, avg_total_loss, avg_model_loss, avg_mask_loss, accuracy)


        print("Training finished.")



    