"""
GraphRAG Explanation service. (Application Layer)
"""

from typing import Dict, Any, Optional
from src.domain.interfaces import ILLMService, IGraphRetriever, IObservabilityService
from src.adapters.logger_adapter import StructuredLogger

logger = StructuredLogger.get_logger(__name__)

class GraphRAGService:
    def __init__(self, llm: ILLMService, graph: IGraphRetriever, obs: Optional[IObservabilityService] = None):
        self.llm = llm
        self.graph = graph
        self.obs = obs

    def explain_prediction(self, transaction_id: str, risk_score: float) -> str:
        logger.info(f"Generating GraphRAG explanation for {transaction_id}")
        
        # 1. Retrieve Context
        context = self.graph.get_transaction_context(transaction_id)
        
        # 2. Build Prompt
        prompt = f"""
        Analyze current transaction for Anti-Money Laundering (AML) risk.
        
        Transaction ID: {transaction_id}
        GNN Risk Score: {risk_score:.4f}
        
        Graph Context (Neo4j Neighborhood):
        {context}
        
        Provide a concise, professional explanation of WHY this transaction is flagged as high risk 
        based on its connections and position in the graph. Answer in Portuguese.
        """
        
        # 3. Generate with LLM
        explanation = self.llm.generate(prompt)
        
        # 4. Log (optional)
        if self.obs:
            self.obs.log_run("GraphRAG_Explanation", {"tx_id": transaction_id}, {"explanation": explanation})
            
        return explanation
