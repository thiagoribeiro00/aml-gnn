"""
GNN Architectures for Transaction Classification.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from typing import Optional

class GraphSAGEModel(torch.nn.Module):
    """
    2-Layer GraphSAGE model for binary classification.
    """
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
    """
    2-Layer Graph Attention Network (GAT) model for binary classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2, heads: int = 8, dropout: float = 0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # On the second layer, we concatenate=False to get average/output
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
