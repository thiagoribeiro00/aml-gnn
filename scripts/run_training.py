"""
Main script to run the GNN training pipeline.
Usage: python scripts/run_training.py [--model sage|gat] [--epochs 100]
"""

import argparse
import os
import sys
import torch
import mlflow

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger
from src.adapters.pytorch_adapter import PyTorchAdapter
from src.use_cases.data_ingestion import DataIngestion
from src.use_cases.feature_engineering import FeatureEngineering
from src.use_cases.train_model import GNNTrainer
from src.models.gnn_architecture import GraphSAGEModel, GATModel

def main():
    parser = argparse.ArgumentParser(description="AML GNN Training Pipeline")
    parser.add_argument("--model", type=str, default="sage", choices=["sage", "gat"], help="GNN architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # 1. Initialize Logger
    StructuredLogger.configure(level="INFO", environment=config.ENVIRONMENT)
    logger = StructuredLogger.get_logger("training_pipeline")
    logger.info("Starting training pipeline", extra={"cli_args": vars(args)})

    # 2. Prepare Data (Ingestion -> Feature Engineering -> PyTorch Adapter)
    try:
        logger.info("Running full data pipeline")
        ingestion = DataIngestion()
        raw_data = ingestion.ingest_raw_data()
        df = ingestion.clean_and_merge(raw_data)
        
        fe = FeatureEngineering()
        df = fe.scale_features(df)
        
        adapter = PyTorchAdapter()
        data = adapter.prepare_graph_data(df, raw_data["edges"])
        
        # Save processed data for inference
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        storage.save_tensor(data.x, os.path.join(config.PROCESSED_DATA_DIR, "x.pt"))
        storage.save_tensor(data.edge_index, os.path.join(config.PROCESSED_DATA_DIR, "edge_index.pt"))
        storage.save_tensor(data.y, os.path.join(config.PROCESSED_DATA_DIR, "y.pt"))
        storage.save_pickle(df, os.path.join(config.PROCESSED_DATA_DIR, "scaled_data.pkl"))
        logger.info("Processed tensors and interim data saved to disk")
    except Exception as e:
        logger.critical("Data pipeline failed", extra={"error": str(e)})
        sys.exit(1)

    # 3. Initialize Model
    in_channels = data.num_node_features
    if args.model == "sage":
        model = GraphSAGEModel(in_channels=in_channels, hidden_channels=args.hidden, out_channels=2)
    else:
        model = GATModel(in_channels=in_channels, hidden_channels=args.hidden, out_channels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. MLflow Experiment Tracking
    tracking_uri = config.MLFLOW_TRACKING_URI
    # Ensure it's treated as a local path if not a remote URL
    if not tracking_uri.startswith(('http://', 'https://')):
        tracking_uri = os.path.abspath(tracking_uri)
        if not tracking_uri.startswith('file://'):
            # On Windows, we need file:///; on Unix, file:// is enough
            # But MLflow handles plain paths well too.
            pass 
            
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("AML_GNN_Detection")

    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))
        mlflow.log_param("device", device)
        
        # 5. Training
        trainer = GNNTrainer(model=model, optimizer=optimizer, device=device, mlflow_run=run)
        trainer.train(data, epochs=args.epochs)
        
        # 6. Final Evaluation on Test Set
        test_metrics = trainer.evaluate(data, "test_mask")
        logger.info("Final Test Metrics", extra={"metrics": test_metrics})
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)
            
        # 7. Save Model
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        save_path = os.path.join(config.MODELS_DIR, f"{args.model}_best.pt")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        
        # 8. Register Model in MLflow Registry
        try:
            model_name = f"AML_GNN_{args.model.upper()}_Model"
            mlflow.pytorch.log_model(
                model, 
                "model", 
                registered_model_name=model_name
            )
            logger.info(f"Model registered in MLflow Registry as {model_name}")
        except Exception as e:
            logger.warning("Failed to register model in MLflow Registry", extra={"error": str(e)})

if __name__ == "__main__":
    main()
