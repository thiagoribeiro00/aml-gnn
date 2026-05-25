"""
GNN Model Architectures and Factory. (Domain Layer)
Simplified and consolidated for better visibility.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from typing import Optional, Dict, Any

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2, dropout: float = 0.5):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2, heads: int = 8, dropout: float = 0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ModelFactory:
    """
    Senior abstraction to decouple model creation from execution scripts.
    """
    @staticmethod
    def create(model_type: str, in_channels: int, hidden_channels: int = 64, out_channels: int = 2) -> torch.nn.Module:
        if model_type.lower() == "sage":
            return GraphSAGEModel(in_channels, hidden_channels, out_channels)
        elif model_type.lower() == "gat":
            return GATModel(in_channels, hidden_channels, out_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def load(model_type: str, path: str, in_channels: int, device: str = "cpu") -> torch.nn.Module:
        model = ModelFactory.create(model_type, in_channels)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.eval()
        return model
