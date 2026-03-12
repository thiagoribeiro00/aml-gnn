"""
Script to run inference and save predictions back to Neo4j and disk.
Usage: python scripts/deploy_predictions.py --model_path models/sage_best.pt
"""

import argparse
import os
import sys
import torch
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger
from src.adapters.pytorch_adapter import PyTorchAdapter
from src.adapters.neo4j_adapter import Neo4jAdapter
from src.use_cases.predict_and_save import InferencePipeline
from src.models.gnn_architecture import GraphSAGEModel, GATModel

def main():
    parser = argparse.ArgumentParser(description="AML GNN Deployment Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--model_type", type=str, default="sage", choices=["sage", "gat"])
    args = parser.parse_args()

    # 1. Initialize Logger
    StructuredLogger.configure(level="INFO", environment=config.ENVIRONMENT)
    logger = StructuredLogger.get_logger("deployment_pipeline")
    logger.info("Starting deployment pipeline", extra={"model": args.model_path})

    # 2. Load Data (Tensors)
    try:
        x = storage.load_tensor(os.path.join(config.PROCESSED_DATA_DIR, "x.pt"))
        edge_index = storage.load_tensor(os.path.join(config.PROCESSED_DATA_DIR, "edge_index.pt"))
        y = storage.load_tensor(os.path.join(config.PROCESSED_DATA_DIR, "y.pt"))
        
        # We need the txIds to map predictions back.
        # These are usually in the merged_data.pkl or interim pkl.
        df_interim = storage.load_pickle(os.path.join(config.PROCESSED_DATA_DIR, "scaled_data.pkl"))
        tx_ids = df_interim["txId"].tolist()
        
    except Exception as e:
        logger.critical("Failed to load required data/tensors", extra={"error": str(e)})
        sys.exit(1)

    # 3. Load Model
    in_channels = x.shape[1]
    if args.model_type == "sage":
        model = GraphSAGEModel(in_channels=in_channels, hidden_channels=64, out_channels=2)
    else:
        model = GATModel(in_channels=in_channels, hidden_channels=64, out_channels=2)

    try:
        model.load_state_dict(torch.load(args.model_path))
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.critical("Failed to load model weights", extra={"error": str(e)})
        sys.exit(1)

    # 4. Run Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = InferencePipeline(model=model, device=device)
    
    # Create Data object mock for pipeline
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    results_df = pipeline.predict(data)
    results_df["tx_id"] = tx_ids
    
    # 5. Save locally
    output_path = os.path.join(config.DATA_DIR, "results", "predictions.csv")
    pipeline.save_predictions(results_df, output_path)
    
    # 6. Save back to Neo4j
    try:
        logger.info("Writing risk scores back to Neo4j...")
        adapter = Neo4jAdapter()
        
        # Batch update risk scores
        query = """
        UNWIND $batch AS row
        MATCH (t:Transaction {tx_id: row.tx_id})
        SET t.risk_score = row.risk_score
        """
        batch_records = results_df[["tx_id", "risk_score"]].to_dict('records')
        
        # Use existing execute_query in a loop for simplicity or add batch method
        # For scores, we'll just use a loop of batches
        batch_size = 5000
        for i in range(0, len(batch_records), batch_size):
            batch = batch_records[i:i + batch_size]
            adapter.execute_query(query, {"batch": batch})
            
        logger.info("Risk scores successfully synced to Neo4j")
    except Exception as e:
        logger.error("Failed to sync scores to Neo4j", extra={"error": str(e)})

if __name__ == "__main__":
    main()
