import pytest
from src.domain.entities import TransactionNode
from src.domain.value_objects import RiskScore

def test_transaction_node_creation():
    node = TransactionNode(tx_id=1, time_step=1, features=[0.1, 0.2], label=0)
    assert node.tx_id == 1
    assert node.label == 0

def test_risk_score_validation():
    score = RiskScore(0.5)
    assert score.value == 0.5
    
    with pytest.raises(ValueError):
        RiskScore(1.5)
    
    with pytest.raises(ValueError):
        RiskScore(-0.1)
