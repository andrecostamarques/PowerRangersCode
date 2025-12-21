"""
Federated Learning Simulation Runner.

This is the main entry point for running federated learning experiments
with learnable masks. It supports all datasets available in DatasetsDict.

Usage:
    python run_simulation.py --dataset mnist --num_clients 5 --num_rounds 10
"""

import sys
import os
import argparse
import warnings
from datetime import datetime

# Suppress Flower deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)

# Add paths to import existing modules - must be done before any other imports
_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
_fl_path = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, _utils_path)
sys.path.insert(0, _models_path)
sys.path.insert(0, _fl_path)

# Set environment variable for Ray workers to find the modules
os.environ['PYTHONPATH'] = f"{_utils_path}:{_models_path}:{_fl_path}:" + os.environ.get('PYTHONPATH', '')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

import flwr as fl
from flwr.common import ndarrays_to_parameters

# Import existing modules
from DatasetsDict import DatasetDict
from LeNet5 import LeNet5
from ResNet20 import resnet20
import SelectionMask as sm

# Import FL modules
from fl_client import FlowerClient, create_client_fn
from fl_strategy import FedAvgWithMask


# Dataset configurations
DATASET_CONFIGS = {
    "mnist": {
        "model_fn": LeNet5,
        "mask_shape": (1, 28, 28),
        "num_classes": 10,
    },
    "fmnist": {
        "model_fn": LeNet5,
        "mask_shape": (1, 28, 28),
        "num_classes": 10,
    },
    "cifar10": {
        "model_fn": lambda: resnet20(num_classes=10),
        "mask_shape": (3, 32, 32),
        "num_classes": 10,
    },
    "svhn": {
        "model_fn": lambda: resnet20(num_classes=10),
        "mask_shape": (3, 32, 32),
        "num_classes": 10,
    },
    "galaxy10": {
        "model_fn": lambda: resnet20(num_classes=10),
        "mask_shape": (3, 256, 256),
        "num_classes": 10,
    },
}


def partition_data_iid(dataset, num_clients: int, seed: int = 42):
    """
    Partition dataset IID among clients.
    
    Args:
        dataset: The full dataset.
        num_clients: Number of clients.
        seed: Random seed.
        
    Returns:
        List of Subset objects, one per client.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    
    # Split evenly
    split_size = len(dataset) // num_clients
    client_subsets = []
    
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < num_clients - 1 else len(dataset)
        client_indices = indices[start_idx:end_idx]
        client_subsets.append(Subset(dataset, client_indices))
    
    return client_subsets


def create_client_data(
    train_dataset,
    test_dataset,
    num_clients: int,
    batch_size: int,
    eval_split: float = 0.1,
    seed: int = 42
):
    """
    Create training and validation DataLoaders for each client.
    
    Args:
        train_dataset: Full training dataset.
        test_dataset: Full test dataset (used for validation).
        num_clients: Number of clients.
        batch_size: Batch size for DataLoaders.
        eval_split: Fraction of each client's data for validation.
        seed: Random seed.
        
    Returns:
        Dictionary mapping client_id to (train_loader, val_loader).
    """
    # Partition training data among clients
    client_train_subsets = partition_data_iid(train_dataset, num_clients, seed)
    
    # Partition test data for client-side validation
    client_val_subsets = partition_data_iid(test_dataset, num_clients, seed + 1)
    
    client_data = {}
    generator = torch.Generator().manual_seed(seed)
    
    for i in range(num_clients):
        train_subset = client_train_subsets[i]
        val_subset = client_val_subsets[i]
        
        # Optionally split training data for local validation
        # (we use test data partition instead for cleaner separation)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False
        )
        
        client_data[i] = (train_loader, val_loader)
    
    return client_data


def get_initial_parameters(model_fn, mask_shape, lambda_init=1.0):
    """
    Get initial parameters for the global model and mask.
    
    Args:
        model_fn: Function to create model instance.
        mask_shape: Shape of the SelectionMask.
        
    Returns:
        Flower Parameters object.
    """
    model = model_fn()
    mask_model = sm.SelectionMask(shape=mask_shape)
    
    model_params = [val.cpu().numpy() for val in model.state_dict().values()]
    mask_params = [val.cpu().numpy() for val in mask_model.state_dict().values()]
    
    return ndarrays_to_parameters(model_params + mask_params + [np.array([lambda_init], dtype=np.float32)]), model, mask_model


def run_simulation(
    dataset_name: str = "mnist",
    num_clients: int = 5,
    num_rounds: int = 10,
    local_epochs: int = 5,
    batch_size: int = 64,
    model_lr: float = 1e-3,
    mask_lr: float = 1e-3,
    lambda_init: float = 1.0,
    lambda_factor: float = 1.5,
    lambda_patience: int = 2,
    lambda_threshold: float = 0.0025,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    seed: int = 42,
    checkpoint_dir: str = "../../checkpoints/federated/",
    experiment_id: str = None,
):
    """
    Run a federated learning simulation.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', etc.).
        num_clients: Number of clients in the federation.
        num_rounds: Number of federated rounds.
        local_epochs: Number of local epochs per round.
        batch_size: Batch size for local training.
        model_lr: Learning rate for model parameters.
        mask_lr: Learning rate for mask parameters.
        lambda_init: Initial lambda for mask regularization.
        lambda_factor: Lambda multiplication factor.
        lambda_patience: Epochs before increasing lambda.
        lambda_threshold: Minimum improvement threshold.
        fraction_fit: Fraction of clients to train per round.
        fraction_evaluate: Fraction of clients to evaluate per round.
        seed: Random seed.
        checkpoint_dir: Directory for checkpoints.
        experiment_id: Unique experiment identifier.
        
    Returns:
        History object with training results.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get dataset configuration
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    
    ds_config = DATASET_CONFIGS[dataset_name]
    model_fn = ds_config["model_fn"]
    mask_shape = ds_config["mask_shape"]
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Federated Learning Simulation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local Epochs: {local_epochs}")
    print(f"Mask Shape: {mask_shape}")
    print(f"{'='*60}\n")
    
    db = DatasetDict()
    ds_list, tf_train, tf_test = db.get(dataset_name)
    train_dataset, test_dataset = ds_list
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Samples per client (approx): {len(train_dataset) // num_clients}")
    
    # Create client data
    client_data = create_client_data(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed
    )
    
    # Client configuration
    client_config = {
        "local_epochs": local_epochs,
        "model_lr": model_lr,
        "mask_lr": mask_lr,
        "lambda_init": lambda_init,
        "lambda_factor": lambda_factor,
        "lambda_patience": lambda_patience,
        "lambda_threshold": lambda_threshold,
    }
    
    # Create client factory function
    client_fn = create_client_fn(
        model_fn=model_fn,
        mask_shape=mask_shape,
        client_data=client_data,
        config=client_config
    )
    
    # Get initial parameters
    initial_parameters, model, mask_model = get_initial_parameters(model_fn, mask_shape, lambda_init)
    
    # Create experiment ID if not provided
    if experiment_id is None:
        experiment_id = f"{dataset_name}_fl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create strategy
    strategy = FedAvgWithMask(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min(2, num_clients),
        min_evaluate_clients=min(2, num_clients),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        checkpoint_dir=checkpoint_dir,
        experiment_id=experiment_id,
        fit_metrics_aggregation_fn=lambda metrics: {},
        evaluate_metrics_aggregation_fn=lambda metrics: {},
    )
    
    # Configure simulation
    # Run clients SEQUENTIALLY (one at a time) to avoid GPU memory conflicts.
    # Each client uses the full GPU, but only one runs at a time.
    if torch.cuda.is_available():
        # Request full GPU per client = only 1 client runs at a time
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}
        print(f"GPU detected: Running clients SEQUENTIALLY (1 at a time) on GPU")
    else:
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}
        print(f"No GPU: Running on CPU")
    
    # Run simulation
    print(f"\nStarting federated training...")
    print(f"Checkpoints will be saved to: {os.path.join(checkpoint_dir, experiment_id)}")
    
    # Configure Ray runtime environment so workers can find our modules
    ray_init_args = {
        "runtime_env": {
            "env_vars": {
                "PYTHONPATH": f"{_utils_path}:{_models_path}:{_fl_path}"
            },
            "py_modules": [_utils_path, _models_path, _fl_path],
        },
        "ignore_reinit_error": True,
    }
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )
    
    # Save final model
    if history.metrics_centralized:
        final_params = strategy.save_final_model(
            ndarrays_to_parameters(
                [val.cpu().numpy() for val in model.state_dict().values()] +
                [val.cpu().numpy() for val in mask_model.state_dict().values()]
            ),
            model,
            mask_model
        )
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Results saved to: {os.path.join(checkpoint_dir, experiment_id)}")
    print(f"{'='*60}")
    
    return history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Learning with Learnable Masks"
    )
    
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use (default: mnist)"
    )
    parser.add_argument(
        "--num_clients", type=int, default=5,
        help="Number of federated clients (default: 5)"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=10,
        help="Number of federated rounds (default: 10)"
    )
    parser.add_argument(
        "--local_epochs", type=int, default=5,
        help="Local epochs per round (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--model_lr", type=float, default=1e-3,
        help="Model learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--mask_lr", type=float, default=1e-3,
        help="Mask learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lambda_init", type=float, default=1.0,
        help="Initial lambda for mask regularization (default: 1.0)"
    )
    parser.add_argument(
        "--lambda_factor", type=float, default=1.5,
        help="Lambda multiplication factor (default: 1.5)"
    )
    parser.add_argument(
        "--lambda_patience", type=int, default=2,
        help="Epochs before increasing lambda (default: 2)"
    )
    parser.add_argument(
        "--lambda_threshold", type=float, default=0.0025,
        help="Minimum improvement threshold (default: 0.0025)"
    )
    parser.add_argument(
        "--fraction_fit", type=float, default=1.0,
        help="Fraction of clients for training per round (default: 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--experiment_id", type=str, default=None,
        help="Experiment identifier (default: auto-generated)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    history = run_simulation(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        model_lr=args.model_lr,
        mask_lr=args.mask_lr,
        lambda_init=args.lambda_init,
        lambda_factor=args.lambda_factor,
        lambda_patience=args.lambda_patience,
        lambda_threshold=args.lambda_threshold,
        fraction_fit=args.fraction_fit,
        seed=args.seed,
        experiment_id=args.experiment_id,
    )
