from abc import ABC, abstractmethod
from typing import List, Any
from .entities import TransactionNode

class GraphRepository(ABC):
    @abstractmethod
    def save_transaction(self, node: TransactionNode) -> None:
        pass

    @abstractmethod
    def get_transaction(self, tx_id: int) -> TransactionNode:
        pass

    @abstractmethod
    def get_neighbors(self, tx_id: int) -> List[TransactionNode]:
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        pass
