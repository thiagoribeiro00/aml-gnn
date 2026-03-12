"""
Feature engineering use case for scaling features and preparing graph data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger, log_execution_time, metrics

logger = StructuredLogger.get_logger(__name__)

class FeatureEngineering:
    """Handles feature scaling and preparing train/val/test splits based on time steps."""

    def __init__(self):
        self.scaler = StandardScaler()

    @log_execution_time(logger)
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the 165 features (excluding txId, timestep, and class).
        """
        # Feature columns are from index 2 to 166 (renamed columns starting at index 1)
        # Actually in our merged_df, txId is index, or just a column.
        # Original features start after 'timestep' and before 'class'.
        
        # Get feature columns (0 to 164 in the original features file, now in merged_df)
        # The columns are likely named sequentially or we can use column indices.
        # merged_df columns: [txId, timestep, 2, 3, ..., 166, class]
        
        feature_cols = [col for col in df.columns if col not in ["txId", "timestep", "class"]]
        
        logger.info("Scaling features", extra={"num_features": len(feature_cols)})
        
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        logger.info("Feature scaling completed")
        return df

    @log_execution_time(logger)
    def temporal_split(self, df: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """
        Split node indices based on time steps.
        Time steps 1-34: Train
        Time steps 35-42: Val
        Time steps 43-49: Test
        """
        # Node indices depend on the order in the dataframe.
        # We need to ensure we return indices that match the tensor order later.
        
        # Get indices for each split
        train_idx = df[df["timestep"] <= 34].index.tolist()
        val_idx = df[(df["timestep"] > 34) & (df["timestep"] <= 42)].index.tolist()
        test_idx = df[df["timestep"] > 42].index.tolist()
        
        metrics.record("train_nodes", len(train_idx))
        metrics.record("val_nodes", len(val_idx))
        metrics.record("test_nodes", len(test_idx))
        
        logger.info("Temporal split completed", extra={
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        })
        
        return train_idx, val_idx, test_idx

if __name__ == "__main__":
    # Quick test if run directly
    try:
        df = storage.load_pickle("data/interim/merged_data.pkl")
        fe = FeatureEngineering()
        df = fe.scale_features(df)
        train, val, test = fe.temporal_split(df)
        storage.save_pickle(df, "data/processed/scaled_data.pkl")
        metrics.log_summary()
    except Exception as e:
        logger.critical("Feature engineering pipeline failed", extra={"error": str(e)})
