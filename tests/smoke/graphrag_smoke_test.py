import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.adapters.llm.gemini_adapter import GeminiAdapter
from src.adapters.graph.neo4j_adapter import GraphRetrieverAdapter
from src.adapters.observability.langsmith_adapter import LangSmithAdapter
from src.application.graphrag.explain_use_case import GenerateExplanationUseCase
from src.adapters.logger_adapter import StructuredLogger

def test_graphrag_smoke():
    StructuredLogger.configure(level="INFO")
    logger = StructuredLogger.get_logger("graphrag_smoke_test")
    
    logger.info("Starting GraphRAG Smoke Test")
    
    try:
        obs = LangSmithAdapter()
        llm = GeminiAdapter()
        retriever = GraphRetrieverAdapter()
        explainer = GenerateExplanationUseCase(llm, retriever, obs)
        
        tx_id = "230490356" 
        score = 0.9876
        
        logger.info(f"Generating explanation for TX: {tx_id}...")
        explanation = explainer.execute(tx_id, score)
        
        print("\n--- GENERATED EXPLANATION ---")
        print(explanation)
        print("-----------------------------\n")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")

if __name__ == "__main__":
    test_graphrag_smoke()
