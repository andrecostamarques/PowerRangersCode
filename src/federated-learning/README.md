# Federated Learning with Learnable Masks

This module implements Federated Learning using [Flower](https://flower.dev/) for training models with learnable `SelectionMask`.

## Overview

The federated learning approach trains both the classification model (LeNet5/ResNet20) and the SelectionMask across multiple clients. Both model and mask parameters are aggregated using FedAvg, allowing clients to collaboratively learn:
1. **Model weights**: For classification
2. **Mask parameters**: For feature selection (global feature importance)

## Files

| File | Description |
|------|-------------|
| `run_simulation.py` | Main entry point for running FL experiments |
| `fl_client.py` | Flower client implementation |
| `fl_strategy.py` | Custom FedAvg strategy with mask support |

## Quick Start

### 1. Install Flower

```bash
pip install flwr
```

### 2. Run a simulation with MNIST

```bash
cd src/federated-learning
python run_simulation.py --dataset mnist --num_clients 5 --num_rounds 10
```

### 3. Change dataset

Simply change the `--dataset` argument:

```bash
# Fashion-MNIST
python run_simulation.py --dataset fmnist --num_clients 5 --num_rounds 10

# CIFAR-10
python run_simulation.py --dataset cifar10 --num_clients 5 --num_rounds 20

# SVHN
python run_simulation.py --dataset svhn --num_clients 5 --num_rounds 20

# Galaxy10
python run_simulation.py --dataset galaxy10 --num_clients 3 --num_rounds 30
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `mnist` | Dataset to use (`mnist`, `fmnist`, `cifar10`, `svhn`, `galaxy10`) |
| `--num_clients` | `5` | Number of federated clients |
| `--num_rounds` | `10` | Number of federated rounds |
| `--local_epochs` | `5` | Local training epochs per round |
| `--batch_size` | `64` | Batch size for training |
| `--model_lr` | `1e-3` | Model learning rate |
| `--mask_lr` | `1e-3` | Mask learning rate |
| `--lambda_init` | `1.0` | Initial lambda for mask regularization |
| `--fraction_fit` | `1.0` | Fraction of clients per round |
| `--seed` | `42` | Random seed |
| `--experiment_id` | auto | Unique experiment identifier |

## Output

Results are saved to `checkpoints/federated/<experiment_id>/`:

```
checkpoints/federated/mnist_fl_20231217_120000/
├── training_log.csv              # Training metrics per round
├── confusion_matrix_round_1.npy  # Aggregated confusion matrix (NumPy)
├── confusion_matrix_round_1.json # Aggregated confusion matrix (JSON)
├── confusion_matrix_round_2.npy
├── ...
└── final_checkpoint.pt           # Final model and mask weights
```

### Loading Results

```python
import numpy as np
import torch

# Load confusion matrix
cm = np.load("checkpoints/federated/exp_id/confusion_matrix_round_10.npy")
accuracy = 100 * cm.diagonal().sum() / cm.sum()
print(f"Accuracy: {accuracy:.2f}%")

# Load final model
checkpoint = torch.load("checkpoints/federated/exp_id/final_checkpoint.pt")
model_state = checkpoint['model_state_dict']
mask_state = checkpoint['mask_state_dict']
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FL Server                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │  FedAvgWithMask Strategy                        │    │
│  │  - Aggregates model + mask parameters           │    │
│  │  - Saves confusion matrices per round           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Client 1      │ │   Client 2      │ │   Client N      │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │   Model     │ │ │ │   Model     │ │ │ │   Model     │ │
│ │ (LeNet/Res) │ │ │ │ (LeNet/Res) │ │ │ │ (LeNet/Res) │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ SelectionMask│ │ │ │ SelectionMask│ │ │ │ SelectionMask│ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ [Local Data]    │ │ [Local Data]    │ │ [Local Data]    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Comparison with Centralized Training

| Aspect | Centralized (`training_loop.py`) | Federated (`run_simulation.py`) |
|--------|----------------------------------|--------------------------------|
| Data | All in one place | Distributed across clients |
| Training | Sequential epochs | Rounds with local epochs |
| Aggregation | N/A | FedAvg for model + mask |
| Output | Single checkpoint per epoch | Aggregated metrics per round |
