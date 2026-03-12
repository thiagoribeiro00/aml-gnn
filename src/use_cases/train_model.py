"""
Training loop implementation for Graph Neural Networks.
Handles training, validation, and metrics logging.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import time
from src.adapters.logger_adapter import StructuredLogger, metrics
from src.models.loss_functions import get_weighted_loss_fn

logger = StructuredLogger.get_logger(__name__)

class GNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        mlflow_run: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mlflow_run = mlflow_run
        self.best_val_f1 = 0.0

    def train_epoch(self, data: Any) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        
        # Calculate loss (only on training mask and known labels)
        loss_fn = get_weighted_loss_fn(data.y, device=self.device)
        
        # Only compute loss for nodes with known labels in train_mask
        train_mask = data.train_mask
        loss = loss_fn(out[train_mask], data.y[train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data: Any, mask_name: str = "val_mask") -> Dict[str, float]:
        self.model.eval()
        data = data.to(self.device)
        
        out = self.model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        
        mask = getattr(data, mask_name)
        y_true = data.y[mask]
        y_pred = preds[mask]
        
        # Calculate metrics (Simple implementation for now)
        correct = (y_pred == y_true).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0
        
        # For AML, we care about Illicit class (1)
        # Confusion matrix for F1 calculation
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            f"{mask_name}_loss": 0.0, # Placeholder for loss
            f"{mask_name}_acc": acc,
            f"{mask_name}_f1": f1,
            f"{mask_name}_precision": precision,
            f"{mask_name}_recall": recall
        }

    def train(self, data: Any, epochs: int = 100) -> None:
        logger.info("Starting training loop", extra={"epochs": epochs, "device": self.device})
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss = self.train_epoch(data)
            duration = (time.time() - start_time) * 1000
            
            # Evaluate on validation set
            val_metrics = self.evaluate(data, "val_mask")
            
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_metrics['val_mask_f1']:.4f}",
                    extra={
                        "epoch": epoch,
                        "loss": loss,
                        "val_f1": val_metrics['val_mask_f1'],
                        "duration_ms": duration
                    }
                )
                
            # MLflow logging
            if self.mlflow_run:
                import mlflow
                mlflow.log_metric("loss", loss, step=epoch)
                for k, v in val_metrics.items():
                    mlflow.log_metric(k, v, step=epoch)

            # Checkpoint
            if val_metrics["val_mask_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_mask_f1"]
                # Save best model logic will go to predict_and_save or handled here
                
        logger.info("Training completed", extra={"best_val_f1": self.best_val_f1})
