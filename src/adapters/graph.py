"""
Neo4j Graph Adapter. (Adapters Layer)
"""

from typing import List, Any, Dict, Optional
from src.domain.interfaces import IGraphRetriever, GraphRepository
from src.domain.entities import TransactionNode
from src.infrastructure.database import neo4j_client
from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger, log_execution_time
import pandas as pd

logger = StructuredLogger.get_logger(__name__)

class Neo4jAdapter(GraphRepository, IGraphRetriever):
    """Unified Neo4j Adapter for Ingestion and RAG."""
    def __init__(self):
        self._client = neo4j_client

    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        with self._client.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def save_transaction(self, node: TransactionNode) -> None:
        query = "MERGE (t:Transaction {tx_id: $tx_id}) SET t.class = $label, t.risk_score = $risk_score"
        self.execute_query(query, {"tx_id": node.tx_id, "label": node.label, "risk_score": node.risk_score})

    def get_context(self, query: str, limit: int = 5) -> str:
        return str(self.execute_query(query))

    def get_transaction_context(self, transaction_id: str) -> str:
        query = """
        MATCH (t:Transaction {tx_id: toInteger($tx_id)})
        OPTIONAL MATCH (t)-[:FLOWS_TO]-(neighbor:Transaction)
        RETURN t.tx_id as id, t.class as class, neighbor.tx_id as nid, neighbor.class as nclass
        LIMIT 10
        """
        results = self.execute_query(query, {"tx_id": transaction_id})
        if not results: return f"Node {transaction_id} not found."
        return "\n".join([f"Path: {r['id']} -> {r['nid']} (Class: {r['nclass']})" for r in results])

    def get_transaction(self, tx_id: int) -> TransactionNode:
        query = "MATCH (t:Transaction {tx_id: $tx_id}) RETURN t"
        res = self.execute_query(query, {"tx_id": tx_id})
        if not res: raise ValueError(f"Transaction {tx_id} not found")
        data = res[0]['t']
        return TransactionNode(tx_id=data['tx_id'], time_step=data['timestep'], features=[], label=data['class'], risk_score=data.get('risk_score'))

    def get_neighbors(self, tx_id: int) -> List[TransactionNode]:
        query = "MATCH (t:Transaction {tx_id: $tx_id})-[:FLOWS_TO]-(n:Transaction) RETURN n"
        results = self.execute_query(query, {"tx_id": tx_id})
        return [TransactionNode(tx_id=r['n']['tx_id'], time_step=r['n']['timestep'], features=[], label=r['n']['class']) for r in results]

    def batch_save_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 5000):
        query = "UNWIND $batch AS row MERGE (t:Transaction {tx_id: row.tx_id}) SET t.timestep = row.timestep, t.class = row.class"
        for i in range(0, len(nodes), batch_size):
            self.execute_query(query, {"batch": nodes[i:i + batch_size]})

    def batch_save_edges(self, edges: List[Dict[str, Any]], batch_size: int = 5000):
        query = """
        UNWIND $batch AS row 
        MATCH (t1:Transaction {tx_id: row.txId1})
        MATCH (t2:Transaction {tx_id: row.txId2})
        MERGE (t1)-[:FLOWS_TO]->(t2)
        """
        for i in range(0, len(edges), batch_size):
            self.execute_query(query, {"batch": edges[i:i + batch_size]})

    def batch_update_predictions(self, predictions_df: pd.DataFrame, batch_size: int = 5000):
        data = [{"tx_id": int(r['tx_id']), "risk_score": float(r['risk_score'])} for _, r in predictions_df.iterrows()]
        query = "UNWIND $batch AS row MATCH (t:Transaction {tx_id: row.tx_id}) SET t.risk_score = row.risk_score"
        for i in range(0, len(data), batch_size):
            self.execute_query(query, {"batch": data[i:i + batch_size]})
