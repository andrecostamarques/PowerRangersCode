"""
Model Testing Module.

This module provides the `ModelTester` class, which facilitates the evaluation
of trained models (classifier + optional mask) on a test dataset. It handles checkpoint
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

    def test(self, model_id, epoch, root_dir_save='../../checkpoints/', use_mask=True):
        """
        Evaluates the classifier from a specific epoch.

        By default (use_mask=True), it applies the static mask loaded from Epoch 1.
        It loads the model architecture from the first epoch's checkpoint and the
        weights from the specified epoch's checkpoint. Then, it runs inference
        on the test set.

        Args:
            model_id (str): The unique identifier of the training run.
            epoch (int): The epoch number to evaluate.
            root_dir_save (str, optional): Root directory containing checkpoints.
                Defaults to '../../checkpoints/'.
            use_mask (bool, optional): Whether to apply the mask model if available.
                Defaults to True.

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
        """
        # 1. Paths
        path_epoch_1 = os.path.abspath(os.path.join(root_dir_save, model_id, 'checkpoint_epoch_1.pt'))
        full_path_target = os.path.abspath(os.path.join(root_dir_save, model_id, f'checkpoint_epoch_{epoch}.pt'))

        # 2. Load Structure (Epoch 1)
        checkpoint_e1 = torch.load(path_epoch_1, map_location=self.device, weights_only=False)
        model = checkpoint_e1['model_obj']
        
        # Only retrieve the mask if the flag is active
        mask_model = None
        if use_mask:
            mask_model = checkpoint_e1.get('mask_model_obj', None)
            if mask_model is None:
                print(f"⚠️ Warning: 'use_mask' is True, but 'mask_model_obj' was not found in {model_id}.")

        # 3. Load Classifier Weights (Epoch X)
        checkpoint_target = torch.load(full_path_target, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint_target['model_state_dict'])
        
        model.to(self.device).eval()
        if mask_model:
            mask_model.to(self.device).eval()

        all_targets, all_predictions, all_probs = [], [], []

        # 4. Inference
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # MASK APPLICATION: Only if use_mask=True AND mask exists
                X_input = mask_model(X) if (use_mask and mask_model) else X
                
                y_pred = model(X_input)
                
                probs = torch.softmax(y_pred, dim=1)
                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 5. Metrics (Macro)
        cm = confusion_matrix(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        return (cm, accuracy * 100, precision * 100, recall * 100, f1 * 100, 
                all_targets, all_probs, all_predictions)