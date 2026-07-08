#!/usr/bin/env python3
"""
Classifier Training Script.
Trains a classifier (ResNet20, LeNet5RGB, or SimpleCNNRGB) on the Galaxy10 dataset.
Supports training either:
1. Normally (without any mask)
2. With a fixed pre-defined mask loaded from a checkpoint
"""

import sys
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from TrainingConfig import TrainingConfig
from DatasetsDict import DatasetDict

from LeNet256 import LeNet256
from ResNet20 import resnet20
from SimpleCNNRGB import SimpleCNNRGB
from torchvision.models import resnet34


class ClassifierTrainer:
    """
    Trainer class to train classifier models normally or with a fixed mask.
    Only optimizes the parameters of the classifier.
    """
    def __init__(self, config, mask_checkpoint_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.seed
        
        # Initializing the dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.config.get_dataloader(self.seed)

        # Initializing the classifier model
        self.model = self.config.model.to(self.device)
        self.mask_model = None

        # Load fixed mask if path is provided
        if mask_checkpoint_path is not None:
            # Try to resolve relative path against the project root if it doesn't exist relative to CWD
            resolved_path = mask_checkpoint_path
            if not os.path.exists(resolved_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                candidate_path = os.path.join(project_root, mask_checkpoint_path)
                if os.path.exists(candidate_path):
                    resolved_path = candidate_path

            if os.path.exists(resolved_path):
                print(f"Loading fixed mask structure and weights using checkpoint path: {resolved_path}")
                checkpoint_epoch_X = torch.load(resolved_path, map_location=self.device, weights_only=False)
                
                # Check if the target checkpoint is self-contained (has the mask model object directly)
                if 'mask_model_obj' in checkpoint_epoch_X:
                    print(f"Loading self-contained mask model and weights directly from checkpoint.")
                    self.mask_model = checkpoint_epoch_X['mask_model_obj'].to(self.device)
                    if 'mask_state_dict' in checkpoint_epoch_X:
                        self.mask_model.load_state_dict(checkpoint_epoch_X['mask_state_dict'])
                else:
                    # Look for checkpoint_epoch_1.pt in the same folder for model structure
                    chkpt_dir = os.path.dirname(os.path.abspath(resolved_path))
                    path_epoch_1 = os.path.join(chkpt_dir, 'checkpoint_epoch_1.pt')
                    
                    if os.path.exists(path_epoch_1):
                        print(f"Loading mask model structure from epoch 1 checkpoint: {path_epoch_1}")
                        checkpoint_epoch_1 = torch.load(path_epoch_1, map_location=self.device, weights_only=False)
                        self.mask_model = checkpoint_epoch_1['mask_model_obj'].to(self.device)
                        
                        print(f"Loading mask state from epoch checkpoint: {resolved_path}")
                        self.mask_model.load_state_dict(checkpoint_epoch_X['mask_state_dict'])
                    else:
                        # Fallback to instantiating new mask model
                        print(f"Warning: Could not find model structure in checkpoint or checkpoint_epoch_1.pt. Instantiating a new SelectionMask.")
                        from SelectionMask import SelectionMask
                        self.mask_model = SelectionMask(shape=self.config.mask_shape).to(self.device)
                        if 'mask_state_dict' in checkpoint_epoch_X:
                            self.mask_model.load_state_dict(checkpoint_epoch_X['mask_state_dict'])
                        elif 'state_dict' in checkpoint_epoch_X:
                            self.mask_model.load_state_dict(checkpoint_epoch_X['state_dict'])
            else:
                raise FileNotFoundError(f"Specified mask checkpoint path does not exist: {mask_checkpoint_path} (also tried resolving against project root: {resolved_path})")

            # Freeze mask parameters completely
            self.mask_model.eval()
            for p in self.mask_model.parameters():
                p.requires_grad = False

        self.criterion = self.config.model_loss_function()

        # Initializing the Optimizer (optimizing only classification model parameters)
        self.optimizer_class = self.config.optimizer_class
        self.optimizer_kwargs = {}

        if self.optimizer_class == optim.SGD:
            self.optimizer_kwargs['momentum'] = 0.9
        if self.optimizer_class == optim.Adam or self.optimizer_class == optim.AdamW:
            self.optimizer_kwargs['amsgrad'] = True

        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.config.model_learning_rate,
            **self.optimizer_kwargs
        )

        self.training_id = self.config.training_id
        root_dir = self.config.root_dir_save
        self.checkpoint_dir = os.path.abspath(os.path.join(root_dir, self.training_id))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0

    def train_epoch(self):
        loader = self.train_loader
        running_loss = 0.0

        for X, y in loader: 
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.mask_model is not None:
                X = self.mask_model(X)

            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(loader)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self):
        loader = self.val_loader
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                if self.mask_model is not None:
                    X = self.mask_model(X)
                    
                y_pred = self.model(X)
                _, predicted = torch.max(y_pred, 1)

                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_predictions)
        accuracy = 100 * cm.diagonal().sum() / cm.sum()
        return cm, accuracy

    def save_checkpoint(self, epoch, cm, avg_loss, accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_loss': avg_loss,
            'total_loss': avg_loss,
            'mask_loss': 0.0,
            'accuracy': accuracy,
            'cm': cm,
            'has_mask': self.mask_model is not None
        }
        if self.mask_model is not None:
            checkpoint['mask_model_obj'] = self.mask_model

        if epoch == 0:
            checkpoint['model_obj'] = self.model

        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt')

    def log_training(self, epoch, avg_loss, accuracy):
        log_file = os.path.join(self.checkpoint_dir, 'training_log.csv')
        data = [
            epoch + 1,
            avg_loss,
            avg_loss,
            0.0,
            accuracy,
            0.0,
            0,
        ]
        header = [
            'epoch', 
            'total_loss', 
            'model_loss', 
            'mask_loss', 
            'val_accuracy', 
            'lambda_value',
            'lambda_pvariance_count',
        ]
        file_exists = os.path.exists(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data)

    def train(self, resume=False):
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
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.start_epoch = checkpoint['epoch']
                        print(f"Successfully loaded checkpoint from epoch {epoch_num}. Resuming training from epoch {self.start_epoch + 1}...")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load checkpoint {ckpt_file}: {e}. Trying previous one...")
                if not loaded:
                    print("No valid checkpoints could be loaded. Starting from scratch.")

        mask_status = "WITH FIXED MASK" if self.mask_model is not None else "WITHOUT MASK (NORMAL)"
        print(f"Starting training ({self.config.n_epochs} epochs) in {self.device} {mask_status}.")
        print(f"Checkpoints will be saved in: {self.checkpoint_dir}")
        
        for epoch in range(self.start_epoch, self.config.n_epochs):
            print(f"\nEpoch: {epoch+1}/{self.config.n_epochs}")

            self.model.train()
            avg_loss = self.train_epoch()

            self.model.eval()
            cm_array, accuracy = self.validate_epoch()

            print(f"Model Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            self.log_training(epoch, avg_loss, accuracy)
            self.save_checkpoint(epoch, cm_array, avg_loss, accuracy)

        print("Training finished.")


def get_model_and_config(model_name):
    if model_name == 'resnet20':
        return resnet20(), nn.CrossEntropyLoss, 1e-4
    elif model_name == 'resnet34':
        model = resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model, nn.CrossEntropyLoss, 1e-4
    elif model_name == 'lenet256':
        return LeNet256(), nn.NLLLoss, 1e-3
    elif model_name == 'simplecnnrgb':
        return SimpleCNNRGB(), nn.NLLLoss, 1e-3
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train classifier models on Galaxy10 with optional fixed mask.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["resnet20", "resnet34", "lenet256", "simplecnnrgb"], 
        help="Specify which classifier model to train."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200,
        help="Number of epochs to train. Default is 200."
    )
    parser.add_argument(
        "--mask_checkpoint", 
        type=str, 
        default=None,
        help="Path to the mask checkpoint. If provided, the classifier will be trained with this frozen mask applied."
    )
    parser.add_argument(
        "--training_id", 
        type=str, 
        default=None,
        help="Custom training ID/name. If not provided, it will auto-generate one."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume training from the latest valid checkpoint."
    )
    args = parser.parse_args()

    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get("galaxy10")

    model, loss_fn, lr = get_model_and_config(args.model)
    
    # Auto-generate a descriptive training_id if none was provided
    if args.training_id is None:
        if args.mask_checkpoint is not None:
            training_id = f"galaxy10_{args.model}_with_mask"
        else:
            training_id = f"galaxy10_{args.model}_normal"
    else:
        training_id = args.training_id

    config_dict = {
        "model": model,
        "n_epochs": args.epochs,
        "batch_size": 64,
        "mask_shape": (3, 256, 256),
        "model_learning_rate": lr,
        "mask_learning_rate": 0.001,
        
        # Unused by classifier training but required by TrainingConfig initializer signature
        "lambda_init": 0.1,
        "lambda_factor": 1.5,
        "lambda_patience": 3,
        "lambda_treshold": 0.0025,
        
        "training_id": training_id,
        "optimizer_class": optim.AdamW,
        "model_loss_function": loss_fn,
        
        "datasets": ds_list,
        "transform_train": tf_train,
        "transform_test": tf_test,
        "eval_split": 0.1,
        "seed": 42,
    }

    config = TrainingConfig(**config_dict)
    trainer = ClassifierTrainer(config, mask_checkpoint_path=args.mask_checkpoint)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
