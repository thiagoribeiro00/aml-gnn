"""
Gemini LLM Adapter. (Adapters Layer)
"""

from typing import Optional, Dict, Any
from src.domain.interfaces import ILLMService
from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger

logger = StructuredLogger.get_logger(__name__)

class GeminiAdapter(ILLMService):
    def __init__(self):
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
        self.model = GenerativeModel(config.LLM_MODEL_NAME)

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text
