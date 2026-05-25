"""
LangSmith Observability Adapter. (Adapters Layer)
"""

from typing import Optional, Dict, Any
from src.domain.interfaces import IObservabilityService
from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger

logger = StructuredLogger.get_logger(__name__)

class LangSmithAdapter(IObservabilityService):
    def __init__(self):
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
        logger.info("LangSmith tracing enabled")

    def log_run(self, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        logger.debug(f"Logging run to LangSmith: {name}")
        # Integration with LangChain callbacks or tracing happens via environment variables
        pass
