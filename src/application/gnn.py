"""
GNN Training and Inference services. (Application Layer)
"""

import torch
import pandas as pd
import numpy as np
from src.adapters.logger_adapter import StructuredLogger, log_execution_time
from src.domain.loss import get_weighted_loss_fn

logger = StructuredLogger.get_logger(__name__)

class GNNService:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: str = "cpu", mlflow_run=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mlflow_run = mlflow_run

    def train_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        
        # Weighted loss to handle imbalance
        loss_fn = get_weighted_loss_fn(data.y, data.train_mask, self.device)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask].to(self.device))
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict_probs(self, data) -> torch.Tensor:
        self.model.eval()
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        return torch.softmax(out, dim=1)[:, 1]

    @torch.no_grad()
    def optimize_threshold(self, data, mask_name="val_mask") -> float:
        """Find the best threshold for F1 on the validation set."""
        mask = getattr(data, mask_name)
        probs = self.predict_probs(data)[mask].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        
        best_threshold = 0.5
        best_f1 = 0
        
        thresholds = np.linspace(0.01, 0.9, 50)
        from sklearn.metrics import f1_score
        for t in thresholds:
            f1 = f1_score(y_true, (probs >= t).astype(int), average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        logger.info(f"Optimized Threshold for {mask_name}: {best_threshold:.4f} | Best F1: {best_f1:.4f}")
        return best_threshold

    @torch.no_grad()
    def evaluate(self, data, mask_name="test_mask", threshold=0.5) -> dict:
        self.model.eval()
        mask = getattr(data, mask_name)
        probs = self.predict_probs(data)[mask].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        y_pred = (probs >= threshold).astype(int)
        
        # Diagnostic metrics
        num_true_pos = int((y_true == 1).sum())
        num_pred_pos = int((y_pred == 1).sum())
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "true_positives": num_true_pos,
            "pred_positives": num_pred_pos,
            "threshold": threshold
        }

    @torch.no_grad()
    def predict(self, data, threshold=0.5) -> pd.DataFrame:
        probs = self.predict_probs(data).cpu().numpy()
        tx_ids = data.tx_id.cpu().numpy() if hasattr(data, "tx_id") else np.arange(len(probs))
        
        return pd.DataFrame({
            "tx_id": tx_ids,
            "risk_score": probs,
            "prediction": (probs >= threshold).astype(int)
        })
