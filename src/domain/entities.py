from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class TransactionNode:
    tx_id: int
    time_step: int
    features: List[float]
    label: Optional[int] = None  # 0: lawful, 1: illicit, 2: unknown
    risk_score: Optional[float] = None

@dataclass
class AccountEdge:
    source_id: int
    target_id: int
    edge_type: str = "FLOWS_TO"
