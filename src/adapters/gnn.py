"""
PyTorch Geometric Graph Adapter. (Adapters Layer)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from src.infrastructure.storage import storage
from src.adapters.logger_adapter import StructuredLogger, log_execution_time, metrics

logger = StructuredLogger.get_logger(__name__)

class PyTorchAdapter:
    @log_execution_time(logger)
    def prepare_graph_data(self, df: pd.DataFrame, edges_df: pd.DataFrame) -> Data:
        nodes = df["txId"].unique()
        node_id_map = {tx_id: i for i, tx_id in enumerate(nodes)}
        
        feature_cols = [col for col in df.columns if col not in ["txId", "timestep", "class"]]
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        y = torch.tensor(df["class"].values, dtype=torch.long)
        
        mask = edges_df["txId1"].isin(node_id_map) & edges_df["txId2"].isin(node_id_map)
        filtered_edges = edges_df[mask]
        
        src = filtered_edges["txId1"].map(node_id_map).values
        dst = filtered_edges["txId2"].map(node_id_map).values
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        
        known_mask = (y != -1)
        train_mask = known_mask & (torch.tensor(df["timestep"].values) <= 34)
        val_mask = known_mask & (torch.tensor(df["timestep"].values) > 34) & (torch.tensor(df["timestep"].values) <= 42)
        test_mask = known_mask & (torch.tensor(df["timestep"].values) > 42)
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data.tx_id = torch.tensor(nodes, dtype=torch.long)
        metrics.record("graph_nodes", data.num_nodes)
        metrics.record("graph_edges", data.num_edges)
        return data
