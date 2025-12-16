# Utilities

This directory contains helper classes and functions used across the project.

## Modules

### `TrainingConfig`
A centralized configuration class that encapsulates all experiment parameters, including hyperparameters, model architectures, dataset settings, and optimization strategies. It also handles the creation of DataLoaders based on the provided settings.

### `StaticMaskTraining`
The main orchestrator for the training process. It manages the initialization of models and optimizers, executes training and validation loops, handles lambda scheduling for sparsity control, logs metrics, and saves checkpoints.

### `TotalLoss`
A composite loss module that combines the primary classification loss (e.g., CrossEntropy) with the sparsity regularization loss from the Selection Mask.
`Total Loss = Model Loss + (Lambda * Mask Loss)`

### `LambdaScheduler`
Implements a dynamic scheduler for the regularization parameter (lambda). It adjusts the penalty weight based on the loss trend—similar to `ReduceLROnPlateau`—to balance model accuracy and mask sparsity during training.

### `DatasetsDict`
A dictionary-like container that manages standard datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). It provides easy access to training and testing splits with predefined normalization and augmentation transformations.

### `ModelTester`
A utility class designed for evaluating trained model checkpoints. It loads the saved state dictionaries for both the classifier and the mask, performs inference on a test set, and computes performance metrics such as the confusion matrix.

## Conventions

All classes and methods follow the same naming convention and the same folder/directory convention.
