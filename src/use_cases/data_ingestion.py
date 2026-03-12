"""
Data ingestion use case for loading and validating the Elliptic dataset.
"""

import pandas as pd
from typing import Tuple, Dict
from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger, log_execution_time, metrics

logger = StructuredLogger.get_logger(__name__)

class DataIngestion:
    """Handles the initial loading and validation of the raw Elliptic CSV files."""

    def __init__(self):
        self.features_path = config.FEATURES_PATH
        self.edges_path = config.EDGES_PATH
        self.classes_path = config.CLASSES_PATH

    @log_execution_time(logger)
    def ingest_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all three raw CSV files and return them as a dictionary of DataFrames.
        """
        logger.info("Starting raw data ingestion")
        
        # Load features
        # Features file has no header. 
        # Column 0: txId, Column 1: time step, Columns 2-166: features
        df_features = storage.load_csv(self.features_path, header=None)
        
        # Load classes
        df_classes = storage.load_csv(self.classes_path)
        
        # Load edges
        df_edges = storage.load_csv(self.edges_path)
        
        # Record metrics
        metrics.record("raw_features_rows", len(df_features))
        metrics.record("raw_classes_rows", len(df_classes))
        metrics.record("raw_edges_rows", len(df_edges))
        
        logger.info("Data ingestion completed successfully", extra={
            "features_rows": len(df_features),
            "classes_rows": len(df_classes),
            "edges_rows": len(df_edges)
        })
        
        return {
            "features": df_features,
            "classes": df_classes,
            "edges": df_edges
        }

    @log_execution_time(logger)
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Perform basic validation on the loaded DataFrames.
        """
        features = data["features"]
        classes = data["classes"]
        edges = data["edges"]
        
        # Check for matching txIds between features and classes
        # features[0] is txId
        unique_features_tx = set(features[0].unique())
        unique_classes_tx = set(classes["txId"].unique())
        
        if unique_features_tx != unique_classes_tx:
            missing_in_classes = unique_features_tx - unique_classes_tx
            missing_in_features = unique_classes_tx - unique_features_tx
            logger.warning("Mismatch between features and classes txIds", extra={
                "missing_in_classes": len(missing_in_classes),
                "missing_in_features": len(missing_in_features)
            })
            # This shouldn't happen with the official dataset, but we log it
        
        # Check for missing values
        for key, df in data.items():
            missing_counts = df.isnull().sum().sum()
            if missing_counts > 0:
                logger.warning(f"Found missing values in {key}", extra={"count": int(missing_counts)})
                return False
        
        logger.info("Data validation passed")
        return True

    @log_execution_time(logger)
    def clean_and_merge(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge features and classes, rename columns, and map labels.
        Labels: '1' -> 1 (illicit), '2' -> 0 (lawful), 'unknown' -> -1
        """
        features = data["features"]
        classes = data["classes"]
        
        # Rename columns for clarity
        # Column 0: txId, Column 1: timestep
        features = features.rename(columns={0: "txId", 1: "timestep"})
        
        # Merge with classes
        merged = pd.merge(features, classes, on="txId")
        
        # Map labels
        # 1: illicit, 2: lawful, unknown: unknown
        label_map = {"1": 1, "2": 0, "unknown": -1}
        merged["class"] = merged["class"].map(label_map)
        
        # Record stats
        illicit_count = (merged["class"] == 1).sum()
        lawful_count = (merged["class"] == 0).sum()
        unknown_count = (merged["class"] == -1).sum()
        
        metrics.record("illicit_nodes", int(illicit_count))
        metrics.record("lawful_nodes", int(lawful_count))
        metrics.record("unknown_nodes", int(unknown_count))
        
        logger.info("Data merging and mapping completed", extra={
            "illicit": int(illicit_count),
            "lawful": int(lawful_count),
            "unknown": int(unknown_count)
        })
        
        return merged

if __name__ == "__main__":
    # Quick test if run directly
    config.validate()
    ingestion = DataIngestion()
    try:
        raw_data = ingestion.ingest_raw_data()
        if ingestion.validate_data(raw_data):
            processed_df = ingestion.clean_and_merge(raw_data)
            storage.save_pickle(processed_df, "data/interim/merged_data.pkl")
            metrics.log_summary()
    except Exception as e:
        logger.critical("Ingestion pipeline failed", extra={"error": str(e)})
