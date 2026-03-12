"""
Verification script for the Phase 2 implementation.
Runs the data pipeline end-to-end and logs stats.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger, metrics
from src.use_cases.data_ingestion import DataIngestion
from src.use_cases.feature_engineering import FeatureEngineering
from src.adapters.pytorch_adapter import PyTorchAdapter

def run_verification():
    # 1. Initialize Logger
    StructuredLogger.configure(level="INFO", environment="dev")
    logger = StructuredLogger.get_logger("verification")
    
    logger.info("Starting Phase 2 verification pipeline")
    
    try:
        # 2. Ingestion
        ingestion = DataIngestion()
        raw_data = ingestion.ingest_raw_data()
        if not ingestion.validate_data(raw_data):
            logger.error("Data validation failed")
            return
        
        df = ingestion.clean_and_merge(raw_data)
        
        # 3. Feature Engineering
        fe = FeatureEngineering()
        df = fe.scale_features(df)
        train, val, test = fe.temporal_split(df)
        
        # 4. PyTorch Adapter
        adapter = PyTorchAdapter()
        data = adapter.prepare_graph_data(df, raw_data["edges"])
        
        # 5. Output Stats
        logger.info("Verification pipeline completed successfully")
        metrics.log_summary()
        
        print("\n--- Summary ---")
        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {data.num_node_features}")
        print(f"Train nodes (known): {data.train_mask.sum().item()}")
        print(f"Val nodes (known): {data.val_mask.sum().item()}")
        print(f"Test nodes (known): {data.test_mask.sum().item()}")
        
    except Exception as e:
        logger.critical("Verification pipeline CRASHED", extra={"error": str(e)})
        raise

if __name__ == "__main__":
    run_verification()
