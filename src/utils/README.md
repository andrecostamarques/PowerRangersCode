# Utilities

This directory contains helper classes and functions used across the project.

## Modules

### `LambdaScheduler`
Implements a scheduler for the regularization parameter (lambda). It adapts the value based on the loss trend, similar to `ReduceLROnPlateau`, allowing for dynamic adjustment of the sparsity penalty intensity.

### `TotalLoss`
A composite loss module that combines the standard model loss (e.g., CrossEntropy) with the sparsity regularization loss from the Selection Mask.
`Total Loss = Model Loss + (Lambda * Mask Loss)`

### `StaticMaskTraining`
A class structure defining the training and validation loops. It serves as a text-based template or skeleton for implementing the specific training logic for each epoch.
