from dataclasses import dataclass

@dataclass(frozen=True)
class TransactionID:
    value: int

@dataclass(frozen=True)
class RiskScore:
    value: float

    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("RiskScore must be between 0.0 and 1.0")
