from typing import List, Any, Dict
from src.domain.repositories import GraphRepository
from src.domain.entities import TransactionNode
from src.infrastructure.database import neo4j_client
from src.adapters.logger_adapter import StructuredLogger, log_execution_time

logger = StructuredLogger.get_logger(__name__)

class Neo4jAdapter(GraphRepository):
    def __init__(self):
        self._client = neo4j_client

    @log_execution_time(logger)
    def save_transaction(self, node: TransactionNode) -> None:
        query = """
        MERGE (t:Transaction {tx_id: $tx_id})
        SET t.time_step = $time_step,
            t.features = $features,
            t.label = $label,
            t.risk_score = $risk_score
        """
        params = {
            "tx_id": node.tx_id,
            "time_step": node.time_step,
            "features": node.features,
            "label": node.label,
            "risk_score": node.risk_score
        }
        self.execute_query(query, params)

    @log_execution_time(logger)
    def get_transaction(self, tx_id: int) -> TransactionNode:
        query = "MATCH (t:Transaction {tx_id: $tx_id}) RETURN t"
        result = self.execute_query(query, {"tx_id": tx_id})
        if not result:
            logger.warning("Transaction not found", extra={"tx_id": tx_id})
            raise ValueError(f"Transaction {tx_id} not found")
        
        node_data = result[0]['t']
        return TransactionNode(
            tx_id=node_data['tx_id'],
            time_step=node_data['time_step'],
            features=node_data['features'],
            label=node_data.get('label'),
            risk_score=node_data.get('risk_score')
        )

    @log_execution_time(logger)
    def get_neighbors(self, tx_id: int) -> List[TransactionNode]:
        query = """
        MATCH (t:Transaction {tx_id: $tx_id})-[:FLOWS_TO]-(neighbor:Transaction)
        RETURN neighbor
        """
        results = self.execute_query(query, {"tx_id": tx_id})
        neighbors = []
        for record in results:
            node_data = record['neighbor']
            neighbors.append(TransactionNode(
                tx_id=node_data['tx_id'],
                time_step=node_data['time_step'],
                features=node_data['features'],
                label=node_data.get('label'),
                risk_score=node_data.get('risk_score')
            ))
        return neighbors

    def execute_query(self, query: str, params: Any = None) -> List[Any]:
        cid = StructuredLogger.generate_correlation_id()
        logger.debug("Executing Cypher query", extra={"cid": cid, "params_keys": list(params.keys()) if params else []})
        with self._client.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def batch_save_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 1000) -> None:
        """Batch save transaction nodes using UNWIND."""
        query = """
        UNWIND $batch AS row
        MERGE (t:Transaction {tx_id: row.tx_id})
        SET t.timestep = row.timestep,
            t.class = row.class,
            t.features = row.features
        """
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            self.execute_query(query, {"batch": batch})
            logger.info(f"Ingested {min(i + batch_size, len(nodes))} nodes")

    def batch_save_edges(self, edges: List[Dict[str, Any]], batch_size: int = 5000) -> None:
        """Batch save edges using UNWIND."""
        query = """
        UNWIND $batch AS row
        MATCH (src:Transaction {tx_id: row.txId1})
        MATCH (dst:Transaction {tx_id: row.txId2})
        MERGE (src)-[:FLOWS_TO]->(dst)
        """
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            self.execute_query(query, {"batch": batch})
            logger.info(f"Ingested {min(i + batch_size, len(edges))} edges")
