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
import DatasetsDict as dt

class ModelTester: 
    """
    A utility class for testing and evaluating trained models.

    This class handles loading model checkpoints, setting up the evaluation environment
    (device), and computing performance metrics like the confusion matrix on a provided
    test dataloader.

    Attributes:
        device (torch.device): The computation device (CPU or CUDA) used for testing.
        batch (int): Batch size used for the test dataloader.
        test_loader (DataLoader): DataLoader for the test dataset.
    """

    def __init__(self, dataset: str, batch=128):
        """
        Initializes the ModelTester.

        Args:
            dataset (str): The name of the dataset to retrieve from DatasetDict (e.g., 'mnist').
            batch (int, optional): The batch size for testing. Defaults to 128.
        """
        # Set the device for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch = batch

        db = dt.DatasetDict()

        ds_list, _, _ = db.get(dataset)

        self.test_loader = torch.utils.data.DataLoader(ds_list[1], batch_size=self.batch, shuffle=False)

    def test(self, model_pth, root_dir_save = '../../checkpoints/'):
        """
        Evaluates a saved model checkpoint on a test dataset.

        Loads the state dictionaries for both the main model and the mask model from
        a specified checkpoint file, performs inference on the test loader, and
        calculates the confusion matrix.

        Args:
            model_pth (str): The relative path to the specific checkpoint file (e.g., 'training_id/checkpoint_epoch_X.pt').
            root_dir_save (str, optional): The root directory containing the checkpoint file.
                Defaults to '../../checkpoints/'.

        Returns:
            tuple: A tuple containing:
                - cm (numpy.ndarray): The confusion matrix of the test results.
                - all_targets (list): List of ground truth labels.
                - all_probs (list): List of predicted probabilities for each class.
                - all_predictions (list): List of predicted class labels.
        """
        # Construct path to the first epoch checkpoint to retrieve model architecture objects
        training_dir = os.path.dirname(model_pth)
        path_epoch_1 = os.path.join(root_dir_save, training_dir, 'checkpoint_epoch_1.pt')
        path_epoch_1 = os.path.abspath(path_epoch_1)

        # Load the architecture objects (Model and Mask) saved in epoch 1
        checkpoint_e1 = torch.load(path_epoch_1, map_location=self.device, weights_only=False)
        model = checkpoint_e1['model_obj']
        mask_model = checkpoint_e1['mask_model_obj']

        # Construct the full path for the target checkpoint
        full_path = os.path.join(root_dir_save, model_pth)
        full_path = os.path.abspath(full_path)

        # Load the state dictionaries into the models
        checkpoint_model = torch.load(full_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint_model['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Load mask state and set to eval mode
        mask_model.load_state_dict(checkpoint_model['mask_state_dict'])
        mask_model.to(self.device)
        mask_model.eval()

        all_targets = []
        all_predictions = []
        all_probs = []

        # Perform inference without gradient calculation
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)

                X_masked = mask_model(X)
                y_pred = model(X_masked)

                probs = torch.softmax(y_pred, dim=1)

                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Generate the Confusion Matrix to return statistical performance data
        cm = confusion_matrix(all_targets, all_predictions)

        return cm, all_targets, all_probs, all_predictions
