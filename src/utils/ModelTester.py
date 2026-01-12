"""
Model Testing Module.

This module provides the `ModelTester` class, which facilitates the evaluation
of trained models (classifier + mask) on a test dataset. It handles checkpoint
loading, inference execution, and metric calculation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import os 
import DatasetsDict as dt

class ModelTester: 
    """
    Evaluates trained models on a specified dataset.

    This class manages the loading of test datasets and performs evaluation
    routines using saved checkpoints. It computes standard classification
    metrics such as accuracy, precision, recall, and F1-score.

    Attributes:
        device (torch.device): The computation device (CPU or CUDA).
        batch (int): Batch size for the data loader.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    def __init__(self, dataset: str, batch=128):
        """
        Initializes the ModelTester.

        Args:
            dataset (str): The name of the dataset to load (e.g., 'galaxy10').
            batch (int, optional): Batch size for testing. Defaults to 128.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch = batch
        db = dt.DatasetDict()
        ds_list, _, _ = db.get(dataset)
        # ds_list[1] is the test set
        self.test_loader = torch.utils.data.DataLoader(ds_list[1], batch_size=self.batch, shuffle=False)

    def test(self, model_id, epoch, root_dir_save='../../checkpoints/'):
        """
        Performs a full test run and returns MACRO metrics.

        Loads the model architecture from the first epoch's checkpoint and the
        weights from the specified epoch's checkpoint. Then, it runs inference
        on the test set.

        Args:
            model_id (str): The unique identifier of the training run.
            epoch (int): The epoch number to evaluate.
            root_dir_save (str, optional): Root directory containing checkpoints.
                Defaults to '../../checkpoints/'.

        Returns:
            tuple: A tuple containing:
                - cm (numpy.ndarray): Confusion matrix.
                - accuracy (float): Global accuracy percentage.
                - precision (float): Macro-averaged precision percentage.
                - recall (float): Macro-averaged recall percentage.
                - f1 (float): Macro-averaged F1-score percentage.
                - targets (list): List of ground truth labels.
                - probs (list): List of predicted probabilities.
                - predictions (list): List of predicted class labels.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        # 1. Dynamic Paths
        path_epoch_1 = os.path.abspath(os.path.join(root_dir_save, model_id, 'checkpoint_epoch_1.pt'))
        filename = f'checkpoint_epoch_{epoch}.pt'
        full_path = os.path.abspath(os.path.join(root_dir_save, model_id, filename))

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Checkpoint not found at: {full_path}")

        # 2. Load Architecture (from epoch 1) and Weights (from desired epoch)
        checkpoint_e1 = torch.load(path_epoch_1, map_location=self.device, weights_only=False)
        model = checkpoint_e1['model_obj']
        mask_model = checkpoint_e1['mask_model_obj']

        checkpoint_model = torch.load(full_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint_model['model_state_dict'])
        mask_model.load_state_dict(checkpoint_model['mask_state_dict'])
        
        model.to(self.device).eval()
        mask_model.to(self.device).eval()

        all_targets = []
        all_predictions = []
        all_probs = []

        # 3. Inference Loop
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Pass through the trained mask and then the classifier
                X_masked = mask_model(X)
                y_pred = model(X_masked)
                
                probs = torch.softmax(y_pred, dim=1)
                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 4. Macro Metrics Calculation
        cm = confusion_matrix(all_targets, all_predictions)
        
        # average='macro' calculates the metric for each class and takes the arithmetic mean
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        # Global Accuracy (Micro) for reference
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        return (cm, accuracy * 100, precision * 100, recall * 100, f1 * 100, 
                all_targets, all_probs, all_predictions)
