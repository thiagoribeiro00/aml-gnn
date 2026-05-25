"""
Domain interfaces (Ports) for Graph and LLM services.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from .entities import TransactionNode

class IGraphRetriever(ABC):
    @abstractmethod
    def get_context(self, query: str, limit: int = 5) -> str: pass
    
    @abstractmethod
    def get_transaction_context(self, transaction_id: str) -> str: pass

class ILLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str: pass

class IObservabilityService(ABC):
    @abstractmethod
    def log_run(self, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None): pass

class GraphRepository(ABC):
    @abstractmethod
    def save_transaction(self, node: TransactionNode) -> None: pass
    
    @abstractmethod
    def get_transaction(self, tx_id: int) -> TransactionNode: pass
    
    @abstractmethod
    def get_neighbors(self, tx_id: int) -> List[TransactionNode]: pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Any = None) -> List[Any]: pass
