"""
Use case for generating predictions (risk scores) using a trained GNN model.
Saves results to disk and optionally to Neo4j.
"""

import torch
import pandas as pd
import os
from typing import Any
from src.adapters.logger_adapter import StructuredLogger, log_execution_time
from src.infrastructure.config import config
from src.infrastructure.storage import storage

logger = StructuredLogger.get_logger(__name__)

class InferencePipeline:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @log_execution_time()
    def predict(self, data: Any) -> pd.DataFrame:
        """Generate risk scores for all nodes."""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            # Use softmax to get probabilities (risk scores)
            probs = torch.softmax(out, dim=1)
            # Assuming class 1 is Illicit, risk score is the probability of class 1
            risk_scores = probs[:, 1].cpu().numpy()
            
        # Create results DataFrame
        # We need the original txIds to map results back
        # In this implementation, we assume the index matches the node order in 'data'
        results = pd.DataFrame({
            "risk_score": risk_scores
        })
        
        logger.info("Predictions generated successfully", extra={"count": len(results)})
        return results

    def save_predictions(self, results: pd.DataFrame, path: str):
        """Save predictions to CSV."""
        storage.save_csv(results, path)
        logger.info(f"Predictions saved to {path}")
