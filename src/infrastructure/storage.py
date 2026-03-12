"""
Infrastructure utility for disk I/O operations with structured logging.
"""

import os
import pandas as pd
import pickle
import torch
from typing import Any, Optional
from src.adapters.logger_adapter import StructuredLogger, log_execution_time

logger = StructuredLogger.get_logger(__name__)

class DiskStorage:
    """Handles reading and writing various data formats to disk."""

    @staticmethod
    @log_execution_time(logger)
    def save_pickle(data: Any, path: str) -> None:
        """Save an object to a pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("Successfully saved pickle file", extra={"path": path})

    @staticmethod
    @log_execution_time(logger)
    def load_pickle(path: str) -> Any:
        """Load an object from a pickle file."""
        if not os.path.exists(path):
            logger.error("Pickle file not found", extra={"path": path})
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info("Successfully loaded pickle file", extra={"path": path})
        return data

    @staticmethod
    @log_execution_time(logger)
    def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
        """Save a DataFrame to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index)
        logger.info("Successfully saved CSV file", extra={"path": path, "rows": len(df)})

    @staticmethod
    @log_execution_time(logger)
    def load_csv(path: str, **kwargs) -> pd.DataFrame:
        """Load a DataFrame from a CSV file."""
        if not os.path.exists(path):
            logger.error("CSV file not found", extra={"path": path})
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info("Successfully loaded CSV file", extra={"path": path, "rows": len(df)})
        return df

    @staticmethod
    @log_execution_time(logger)
    def save_tensor(tensor: torch.Tensor, path: str) -> None:
        """Save a PyTorch tensor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path)
        logger.info("Successfully saved tensor", extra={"path": path, "shape": list(tensor.shape)})

    @staticmethod
    @log_execution_time(logger)
    def load_tensor(path: str) -> torch.Tensor:
        """Load a PyTorch tensor from disk."""
        if not os.path.exists(path):
            logger.error("Tensor file not found", extra={"path": path})
            raise FileNotFoundError(f"Tensor file not found: {path}")
        tensor = torch.load(path)
        logger.info("Successfully loaded tensor", extra={"path": path, "shape": list(tensor.shape)})
        return tensor

storage = DiskStorage()
