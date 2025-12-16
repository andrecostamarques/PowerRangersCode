"""
Model evaluation module.

This module provides functionality to load trained model checkpoints (including
both the classification model and the selection mask) and evaluate their performance
on a test dataset using confusion matrices.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os 

class ModelTester: 
    """
    A utility class for testing and evaluating trained models.

    This class handles loading model checkpoints, setting up the evaluation environment
    (device), and computing performance metrics like the confusion matrix on a provided
    test dataloader.

    Attributes:
        device (torch.device): The computation device (CPU or CUDA) used for testing.
    """

    def __init__(self, device='cpu'):
        """
        Initializes the ModelTester.

        Args:
            device (str, optional): The device to run the test on ('cpu' or 'cuda').
                Defaults to 'cpu'.
        """
        # It have the device as variable to make easily to access

        self.device = torch.device(device)

    def test(self, model_pth, model, mask_model, test_loader, root_dir_save = '../../checkpoints/'):
        """
        Evaluates a saved model checkpoint on a test dataset.

        Loads the state dictionaries for both the main model and the mask model from
        a specified checkpoint file, performs inference on the test loader, and
        calculates the confusion matrix.

        Args:
            model_pth (str): The relative path or filename of the checkpoint file.
            model (torch.nn.Module): The instantiated classification model architecture.
            mask_model (torch.nn.Module): The instantiated mask model architecture.
            test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
            root_dir_save (str, optional): The root directory containing the checkpoint file.
                Defaults to '../../checkpoints/'.

        Returns:
            numpy.ndarray: The confusion matrix of the test results.
        """
        # Setting the path for the PTH
        full_path = os.path.join(root_dir_save, model_pth)
        full_path = os.path.abspath(full_path)

        # Loading the PTH
        checkpoint_model = torch.load(full_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint_model['model_state_dict'])
        model.to(self.device)
        model.eval()

        #Loading the models and setting them as eval()
        mask_model.load_state_dict(checkpoint_model['mask_state_dict'])
        mask_model.to(self.device)
        mask_model.eval()

        all_targets = []
        all_predictions = []

        # Running without generating the gradients 
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)

                X_masked = mask_model(X)
                y_pred = model(X_masked)

                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Generating the Confusion Matrix, this way, we return all the statistical data needed

        cm = confusion_matrix(all_targets, all_predictions)

        return cm
