# PowerRangersCode
![image](https://github.com/user-attachments/assets/e8dd40be-5709-4b1d-952f-c5ab8f50c1ee)

## Dependencies Instalation
As Cuda and Torch cannot be installed via the enviroment.yml. The following code should be used.

```bash
conda env create -n PowerRanger-env python=3.12
conda activate PowerRanger-env
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda env update --file environment.yml --prune
```

## Overview
PowerRangersCode is a project enabling Neural Network Training with Learnable Selection Masks. It implements a custom module using the Straight-Through Estimator (STE) to learn hard binary masks for feature selection or pruning.

## Repository Structure

- **`src/`**: Contains the source code for the project, including models, training scripts, and utilities.
- **`notebooks/`**: Contains Jupyter notebooks for running experiments and analyzing results.
- **`data/`**: Directory for storing datasets.

## Checkpoints
All model checkpoints (.pth) will be stored on a soft-link `checkpoints`. You may choose where this link points to.
