import argparse
import os
import sys
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.adapters.gnn import PyTorchAdapter
from src.application.gnn import GNNService
from src.domain.models import ModelFactory
from src.adapters.graph import Neo4jAdapter

def main():
    parser = argparse.ArgumentParser(description="AML GNN Prediction")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="sage")
    args = parser.parse_args()

    # 1. Load and Prepare Data (Ensuring ID alignment)
    from src.application.data import DataService
    data_svc = DataService()
    df = data_svc.clean_and_merge(data_svc.ingest_raw_data())
    df = data_svc.scale_features(df)
    
    adapter_pt = PyTorchAdapter()
    edges_df = pd.read_csv(config.EDGES_FILE)
    data = adapter_pt.prepare_graph_data(df, edges_df)

    # 2. Load Model via Factory
    model = ModelFactory.load(args.model_type, args.model_path, in_channels=data.num_node_features)
    
    # 3. Predict
    service = GNNService(model)
    results = service.predict(data)
    
    # 4. Save & Sync
    output_path = os.path.join(config.DATA_DIR, "results", "predictions.csv")
    results.to_csv(output_path, index=False)
    
    adapter = Neo4jAdapter()
    adapter.batch_update_predictions(results)
    print(f"Predictions saved to {output_path} and synced to Neo4j.")

if __name__ == "__main__":
    main()
