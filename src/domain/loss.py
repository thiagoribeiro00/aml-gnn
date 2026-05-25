"""
Custom loss functions. (Domain Layer)
"""

import torch
from typing import Optional

def get_weighted_loss_fn(y: torch.Tensor, mask: Optional[torch.Tensor] = None, device: str = "cpu") -> torch.nn.Module:
    target_y = y[mask] if mask is not None else y
    known_y = target_y[target_y != -1]
    
    num_total = len(known_y)
    num_illicit = (known_y == 1).sum().item()
    num_lawful = (known_y == 0).sum().item()
    
    weight_illicit = num_total / (2.0 * num_illicit) if num_illicit > 0 else 1.0
    weight_lawful = num_total / (2.0 * num_lawful) if num_lawful > 0 else 1.0
    
    weights = torch.tensor([weight_lawful, weight_illicit], dtype=torch.float).to(device)
    return torch.nn.CrossEntropyLoss(weight=weights)
