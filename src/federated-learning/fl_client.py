"""
Federated Learning with Flower - Client Implementation.

This module implements a Flower client that trains a model with a learnable
SelectionMask locally, following the same training logic as StaticMaskTraining.
"""

import sys
import os

# Add paths to import existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# Import existing modules (without modifying them)
from TotalLoss import TotalLoss
from LambdaScheduler import LambdaScheduler
import SelectionMask as sm


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated training with learnable masks.
    
    This client wraps the model and mask training logic, exposing the
    get_parameters, set_parameters, fit, and evaluate methods required
    by the Flower framework.
    
    Attributes:
        model (nn.Module): The classification model (e.g., LeNet5, ResNet20).
        mask_model (nn.Module): The SelectionMask module.
        train_loader (DataLoader): DataLoader for local training data.
        val_loader (DataLoader): DataLoader for local validation data.
        config (dict): Configuration dictionary with hyperparameters.
        device (torch.device): Device to run computations on.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mask_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        client_id: int = 0
    ):
        """
        Initialize the Flower client.
        
        Args:
            model: The classification neural network.
            mask_model: The SelectionMask module.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Configuration dictionary containing:
                - local_epochs: Number of epochs per round
                - model_lr: Model learning rate
                - mask_lr: Mask learning rate
                - lambda_init, lambda_factor, lambda_patience, lambda_threshold
            client_id: Unique identifier for this client.
        """
        self.model = model
        self.mask_model = mask_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.model.to(self.device)
        self.mask_model.to(self.device)
        
        # Initialize loss and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_scheduler = LambdaScheduler(
            init=config.get('lambda_init', 1.0),
            factor=config.get('lambda_factor', 1.5),
            patience=config.get('lambda_patience', 2),
            treshold=config.get('lambda_threshold', 0.0025)
        )
        
        self.total_loss_calculator = TotalLoss(
            model_loss=self.criterion,
            mask_loss_function=sm.mask_l1_loss,
            lambda_scheduler=self.lambda_scheduler
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': config.get('model_lr', 1e-3)},
            {'params': self.mask_model.parameters(), 'lr': config.get('mask_lr', 1e-3)}
        ], amsgrad=True)
    
    def get_parameters(self, config=None):
        """
        Return model and mask parameters as a list of NumPy arrays.
        
        The parameters are ordered as: [model_params..., mask_params..., lambda_param]
        """
        model_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        mask_params = [val.cpu().numpy() for _, val in self.mask_model.state_dict().items()]
        
        # Add lambda as a 1-element array
        lambda_param = [np.array([self.lambda_scheduler.lbd], dtype=np.float32)]
        
        return model_params + mask_params + lambda_param
    
    def set_parameters(self, parameters):
        """
        Set model and mask parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays in the same order as get_parameters.
        """
        # Extract lambda (last parameter)
        if len(parameters) > 0:
            lambda_param = parameters[-1]
            # Update local lambda scheduler with global value
            self.lambda_scheduler.lbd = float(lambda_param[0])
            
            # Remove lambda from parameters list for model loading
            weights = parameters[:-1]
        else:
            weights = parameters

        # Count model parameters
        model_keys = list(self.model.state_dict().keys())
        mask_keys = list(self.mask_model.state_dict().keys())
        
        num_model_params = len(model_keys)
        
        # Split parameters
        model_params = weights[:num_model_params]
        mask_params = weights[num_model_params:]
        
        # Set model parameters
        model_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(model_keys, model_params)
        })
        self.model.load_state_dict(model_state_dict, strict=True)
        
        # Set mask parameters
        mask_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(mask_keys, mask_params)
        })
        self.mask_model.load_state_dict(mask_state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model and mask locally.
        
        Args:
            parameters: Global parameters from server.
            config: Configuration from server (may override local settings).
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics_dict).
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Get local epochs from server config or use default
        local_epochs = config.get('local_epochs', self.config.get('local_epochs', 1))
        
        # Training loop
        self.model.train()
        self.mask_model.train()
        
        total_loss_sum = 0.0
        model_loss_sum = 0.0
        mask_loss_sum = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with mask
                X_masked = self.mask_model(X)
                y_pred = self.model(X_masked)
                
                # Compute loss
                model_loss, mask_loss, total_loss = self.total_loss_calculator(
                    pred=y_pred,
                    target=y,
                    mask_model=self.mask_model
                )
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                total_loss_sum += total_loss.item()
                model_loss_sum += model_loss.item()
                mask_loss_sum += mask_loss.item()
                num_batches += 1
            
            # Update lambda scheduler after each epoch
            avg_loss = total_loss_sum / num_batches if num_batches > 0 else 0
            self.lambda_scheduler.adapt_lambda(avg_loss)
        
        # Compute metrics
        metrics = {
            "avg_total_loss": total_loss_sum / num_batches if num_batches > 0 else 0,
            "avg_model_loss": model_loss_sum / num_batches if num_batches > 0 else 0,
            "avg_mask_loss": mask_loss_sum / num_batches if num_batches > 0 else 0,
            "lambda": self.lambda_scheduler.lbd,
            "client_id": self.client_id,
        }
        
        return self.get_parameters(), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Global parameters from server.
            config: Configuration from server.
            
        Returns:
            Tuple of (loss, num_samples, metrics_dict).
        """
        self.set_parameters(parameters)
        
        self.model.eval()
        self.mask_model.eval()
        
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                X_masked = self.mask_model(X)
                y_pred = self.model(X_masked)
                
                # Compute loss
                loss = self.criterion(y_pred, y)
                total_loss += loss.item() * len(y)
                
                # Get predictions
                _, predicted = torch.max(y_pred, 1)
                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        accuracy = 100 * cm.diagonal().sum() / cm.sum()
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        # Note: confusion_matrix is NOT returned in metrics because Flower
        # doesn't support nested lists. Instead, we return the flat predictions
        # that the strategy can use to compute the aggregated confusion matrix.
        metrics = {
            "accuracy": float(accuracy),
            "client_id": self.client_id,
            "num_correct": int(cm.diagonal().sum()),
            "num_total": int(cm.sum()),
        }
        
        return avg_loss, len(self.val_loader.dataset), metrics


def create_client_fn(
    model_fn,
    mask_shape: tuple,
    client_data: dict,
    config: dict
):
    """
    Factory function to create FlowerClient instances.
    
    Args:
        model_fn: Function that returns a new model instance.
        mask_shape: Shape of the SelectionMask.
        client_data: Dictionary mapping client_id to (train_loader, val_loader).
        config: Configuration dictionary.
        
    Returns:
        A function that creates a FlowerClient for a given client_id.
    """
    def client_fn(cid: str):
        client_id = int(cid)
        train_loader, val_loader = client_data[client_id]
        
        model = model_fn()
        mask_model = sm.SelectionMask(shape=mask_shape)
        
        return FlowerClient(
            model=model,
            mask_model=mask_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            client_id=client_id
        )
    
    return client_fn
