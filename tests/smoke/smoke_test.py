from src.domain.entities import TransactionNode
from src.domain.value_objects import RiskScore

def test():
    print("Testing TransactionNode...")
    node = TransactionNode(tx_id=1, time_step=1, features=[0.1, 0.2], label=0)
    assert node.tx_id == 1
    assert node.label == 0
    print("TransactionNode OK")

    print("Testing RiskScore...")
    score = RiskScore(0.5)
    assert score.value == 0.5
    print("RiskScore OK")

    try:
        RiskScore(1.5)
        print("RiskScore validation FAILED (should have raised ValueError)")
    except ValueError as e:
        print(f"RiskScore validation OK: {e}")

if __name__ == "__main__":
    test()
