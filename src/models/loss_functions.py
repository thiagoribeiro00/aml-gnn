"""
Custom loss functions for handling class imbalance.
"""

import torch
import torch.nn.functional as F

def get_weighted_loss_fn(y: torch.Tensor, device: str = "cpu") -> torch.nn.Module:
    """
    Calculate weights for CrossEntropyLoss based on class frequencies in y.
    Only considers labels 0 and 1.
    """
    # Count only known labels 0 and 1
    known_y = y[y != -1]
    
    num_total = len(known_y)
    num_illicit = (known_y == 1).sum().item()
    num_lawful = (known_y == 0).sum().item()
    
    # Weight = total / (num_classes * count)
    weight_illicit = num_total / (2.0 * num_illicit) if num_illicit > 0 else 1.0
    weight_lawful = num_total / (2.0 * num_lawful) if num_lawful > 0 else 1.0
    
    weights = torch.tensor([weight_lawful, weight_illicit], dtype=torch.float).to(device)
    
    return torch.nn.CrossEntropyLoss(weight=weights)
