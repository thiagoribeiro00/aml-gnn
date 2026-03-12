"""
Adapter to convert processed DataFrames and edge lists into PyTorch Geometric Data objects.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, List
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger, log_execution_time, metrics

logger = StructuredLogger.get_logger(__name__)

class PyTorchAdapter:
    """Handles conversion from tabular data to graph tensors."""

    @log_execution_time(logger)
    def prepare_graph_data(self, df: pd.DataFrame, edges_df: pd.DataFrame) -> Data:
        """
        Convert node features, labels, and edge list into a torch_geometric.data.Data object.
        """
        logger.info("Preparing graph data tensors")
        
        # 1. Map txIds to sequential indices (0, 1, 2, ...)
        # This is CRITICAL for torch_geometric edge_index
        nodes = df["txId"].unique()
        node_id_map = {tx_id: i for i, tx_id in enumerate(nodes)}
        
        # 2. Extract Features (x)
        # Assuming df is already filtered to only include node features and class
        feature_cols = [col for col in df.columns if col not in ["txId", "timestep", "class"]]
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        
        # 3. Extract Labels (y)
        # Labels are 0 (lawful), 1 (illicit), -1 (unknown)
        y = torch.tensor(df["class"].values, dtype=torch.long)
        
        # 4. Extract Edges (edge_index)
        # We only keep edges where both nodes are in our node set (especially important if we subset)
        # Filter edges where both source and target exist in our node_id_map
        mask = edges_df["txId1"].isin(node_id_map) & edges_df["txId2"].isin(node_id_map)
        filtered_edges = edges_df[mask]
        
        src = filtered_edges["txId1"].map(node_id_map).values
        dst = filtered_edges["txId2"].map(node_id_map).values
        
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        
        # 5. Create Masks for Training (only for known labels)
        # illicit (1) and lawful (0) are "known". unknown (-1) is "unknown".
        known_mask = (y != -1)
        
        # 6. Create Temporal Splits
        train_mask = known_mask & (torch.tensor(df["timestep"].values) <= 34)
        val_mask = known_mask & (torch.tensor(df["timestep"].values) > 34) & (torch.tensor(df["timestep"].values) <= 42)
        test_mask = torch.tensor(df["timestep"].values) > 42 # Test includes unknown for evaluation if needed, but normally we only eval on known labels in test set too
        
        # Refining test mask to only include known labels (consistent with paper)
        test_mask = known_mask & test_mask
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        metrics.record("graph_nodes", data.num_nodes)
        metrics.record("graph_edges", data.num_edges)
        metrics.record("train_mask_size", int(train_mask.sum()))
        
        logger.info("Data object created successfully", extra={
            "nodes": data.num_nodes,
            "edges": data.num_edges,
            "features": data.num_node_features
        })
        
        return data

if __name__ == "__main__":
    # Test stub
    try:
        from src.infrastructure.config import config
        df = storage.load_pickle("data/processed/scaled_data.pkl")
        edges_df = storage.load_csv(config.EDGES_PATH)
        adapter = PyTorchAdapter()
        data = adapter.prepare_graph_data(df, edges_df)
        storage.save_tensor(data.x, "data/processed/x.pt")
        storage.save_tensor(data.edge_index, "data/processed/edge_index.pt")
        storage.save_tensor(data.y, "data/processed/y.pt")
        metrics.log_summary()
    except Exception as e:
        logger.critical("PyTorch adapter failed", extra={"error": str(e)})
