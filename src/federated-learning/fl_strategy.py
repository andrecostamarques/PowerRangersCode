"""
Federated Learning with Flower - Custom Strategy.

This module implements a custom FedAvg strategy that handles both model
and mask parameters, with support for saving confusion matrices and metrics.
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgWithMask(FedAvg):
    """
    Custom FedAvg strategy for aggregating both model and mask parameters.
    
    This strategy extends FedAvg to:
    1. Aggregate model parameters (weights, biases)
    2. Aggregate mask parameters (learns global feature importance)
    3. Save confusion matrices and metrics after each round
    
    Attributes:
        checkpoint_dir (str): Directory to save checkpoints and metrics.
        experiment_id (str): Unique identifier for this experiment.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "../../checkpoints/federated/",
        experiment_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the FedAvgWithMask strategy.
        
        Args:
            checkpoint_dir: Directory for saving results.
            experiment_id: Unique experiment identifier.
            **kwargs: Additional arguments for FedAvg.
        """
        super().__init__(**kwargs)
        
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory
        self.run_dir = os.path.join(checkpoint_dir, self.experiment_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.round_metrics = []
        self.confusion_matrices = []
        self.last_fit_metrics = {}  # Store metrics from aggregate_fit
        
        # Log file
        self.log_file = os.path.join(self.run_dir, "training_log.csv")
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize the CSV log file with headers."""
        headers = [
            'round',
            'num_clients',
            'avg_accuracy',
            'avg_loss',
            'avg_model_loss',
            'avg_mask_loss',
            'avg_lambda',
        ]
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model and mask parameters from clients.
        
        This method uses weighted averaging (FedAvg) for both model
        and mask parameters.
        
        Args:
            server_round: Current round number.
            results: List of (client, fit_result) tuples.
            failures: List of failed operations.
            
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics).
        """
        if not results:
            return None, {}
        
        # Call parent's aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Extract and aggregate metrics from all clients
        total_loss = 0.0
        model_loss = 0.0
        mask_loss = 0.0
        lambda_sum = 0.0
        total_samples = 0
        
        for _, fit_res in results:
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics
            
            total_loss += metrics.get("avg_total_loss", 0) * num_samples
            model_loss += metrics.get("avg_model_loss", 0) * num_samples
            mask_loss += metrics.get("avg_mask_loss", 0) * num_samples
            lambda_sum += metrics.get("lambda", 1.0)
            total_samples += num_samples
        
        if total_samples > 0:
            aggregated_metrics["avg_total_loss"] = total_loss / total_samples
            aggregated_metrics["avg_model_loss"] = model_loss / total_samples
            aggregated_metrics["avg_mask_loss"] = mask_loss / total_samples
            aggregated_metrics["avg_lambda"] = lambda_sum / len(results)
        
        # Store metrics for use in aggregate_evaluate
        self.last_fit_metrics = {
            'avg_total_loss': aggregated_metrics.get('avg_total_loss', 0),
            'avg_model_loss': aggregated_metrics.get('avg_model_loss', 0),
            'avg_mask_loss': aggregated_metrics.get('avg_mask_loss', 0),
            'avg_lambda': aggregated_metrics.get('avg_lambda', 0),
        }
        
        print(f"\n[Round {server_round}] Training completed - "
              f"Total Loss: {self.last_fit_metrics['avg_total_loss']:.4f}, "
              f"Model Loss: {self.last_fit_metrics['avg_model_loss']:.4f}, "
              f"Mask Loss: {self.last_fit_metrics['avg_mask_loss']:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number.
            results: List of (client, evaluate_result) tuples.
            failures: List of failed operations.
            
        Returns:
            Tuple of (aggregated_loss, aggregated_metrics).
        """
        if not results:
            return None, {}
        
        # Aggregate losses and accuracy
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_count = 0
        
        for _, eval_res in results:
            num_samples = eval_res.num_examples
            total_loss += eval_res.loss * num_samples
            total_samples += num_samples
            
            metrics = eval_res.metrics
            total_correct += metrics.get("num_correct", 0)
            total_count += metrics.get("num_total", num_samples)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = 100 * total_correct / total_count if total_count > 0 else 0
        
        # Log metrics
        self._log_round(server_round, len(results), avg_accuracy, avg_loss)
        
        # Save round results
        self._save_round_results(server_round, avg_accuracy, avg_loss, total_correct, total_count)
        
        print(f"[Round {server_round}] Evaluation - "
              f"Accuracy: {avg_accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return avg_loss, {"accuracy": avg_accuracy}
    
    def _save_confusion_matrix(self, round_num: int, cm: np.ndarray):
        """Save confusion matrix for a round."""
        cm_file = os.path.join(self.run_dir, f"confusion_matrix_round_{round_num}.npy")
        np.save(cm_file, cm)
        
        # Also save as JSON for easy viewing
        cm_json = os.path.join(self.run_dir, f"confusion_matrix_round_{round_num}.json")
        with open(cm_json, 'w') as f:
            json.dump({
                "round": round_num,
                "confusion_matrix": cm.tolist(),
                "accuracy": float(100 * cm.diagonal().sum() / cm.sum())
            }, f, indent=2)
    
    def _save_round_results(self, round_num: int, accuracy: float, loss: float, 
                             num_correct: int, num_total: int):
        """Save round evaluation results to JSON."""
        results_file = os.path.join(self.run_dir, f"results_round_{round_num}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "round": round_num,
                "accuracy": accuracy,
                "loss": loss,
                "num_correct": num_correct,
                "num_total": num_total,
            }, f, indent=2)
    
    def _log_round(self, round_num: int, num_clients: int, accuracy: float, loss: float):
        """Log round metrics to CSV."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                num_clients,
                f"{accuracy:.4f}",
                f"{loss:.6f}",
                f"{self.last_fit_metrics.get('avg_model_loss', 0):.6f}",
                f"{self.last_fit_metrics.get('avg_mask_loss', 0):.6f}",
                f"{self.last_fit_metrics.get('avg_lambda', 0):.4f}",
            ])
    
    def save_final_model(self, parameters: Parameters, model, mask_model):
        """
        Save the final aggregated model and mask.
        
        Args:
            parameters: Final aggregated parameters.
            model: Model instance (for architecture reference).
            mask_model: Mask model instance.
        """
        import torch
        
        # Helper to convert parameters to ndarrays
        weights = parameters_to_ndarrays(parameters)
        
        # Extract lambda (last parameter)
        if len(weights) > 0:
            final_lambda = float(weights[-1][0])
            weights = weights[:-1]
        else:
            final_lambda = 0.0

        # Split parameters
        model_keys = list(model.state_dict().keys())
        mask_keys = list(mask_model.state_dict().keys())
        
        num_model_params = len(model_keys)
        model_params = weights[:num_model_params]
        mask_params = weights[num_model_params:]
        
        # Create state dicts
        model_state = {k: torch.tensor(v) for k, v in zip(model_keys, model_params)}
        mask_state = {k: torch.tensor(v) for k, v in zip(mask_keys, mask_params)}
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_state,
            'mask_state_dict': mask_state,
            'model_obj': model,
            'mask_model_obj': mask_model,
            'final_lambda': final_lambda,
        }
        
        checkpoint_path = os.path.join(self.run_dir, "final_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Final model saved to: {checkpoint_path}")
