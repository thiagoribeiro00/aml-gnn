"""
Data Ingestion and Feature Engineering services. (Application Layer)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger, log_execution_time

logger = StructuredLogger.get_logger(__name__)

class DataService:
    @log_execution_time(logger)
    def ingest_raw_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Ingesting raw data from CSV")
        
        # Nodes File (Features) often has no header in the Elliptic dataset
        nodes_df = pd.read_csv(config.NODES_FILE, header=None)
        # Assuming col 0 is txId and col 1 is timestep
        cols = ["txId", "timestep"] + [f"feat_{i}" for i in range(len(nodes_df.columns) - 2)]
        nodes_df.columns = cols
        
        edges_df = pd.read_csv(config.EDGES_FILE)
        classes_df = pd.read_csv(config.CLASSES_FILE)
        return {"nodes": nodes_df, "edges": edges_df, "classes": classes_df}

    @log_execution_time(logger)
    def clean_and_merge(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        nodes = data["nodes"]
        classes = data["classes"]
        df = pd.merge(nodes, classes, on="txId")
        df["class"] = df["class"].apply(lambda x: 1 if x == "1" else (0 if x == "2" else -1))
        return df

    @log_execution_time(logger)
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Scaling node features")
        feature_cols = [col for col in df.columns if col not in ["txId", "timestep", "class"]]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df
