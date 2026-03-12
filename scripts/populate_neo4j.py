"""
Script to populate Neo4j AuraDB with the Elliptic dataset.
Handles batch ingestion of nodes and edges.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger, log_execution_time
from src.adapters.neo4j_adapter import Neo4jAdapter
from src.use_cases.data_ingestion import DataIngestion

def prepare_node_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts for Neo4j."""
    # We exclude the 'class' and 'timestep' from features list if we want them as properties
    feature_cols = [col for col in df.columns if col not in ["txId", "timestep", "class"]]
    
    nodes = []
    for _, row in df.iterrows():
        nodes.append({
            "tx_id": int(row["txId"]),
            "timestep": int(row["timestep"]),
            "class": int(row["class"]),
            "features": row[feature_cols].values.tolist()
        })
    return nodes

@log_execution_time()
def main():
    StructuredLogger.configure(level="INFO", environment=config.ENVIRONMENT)
    logger = StructuredLogger.get_logger("populate_neo4j")
    
    logger.info("Starting Neo4j population process")
    
    try:
        # 1. Load Data
        ingestion = DataIngestion()
        raw_data = ingestion.ingest_raw_data()
        df = ingestion.clean_and_merge(raw_data)
        edges_df = raw_data["edges"]
        
        # 2. Limit data if on Free Tier (Optional, safety check)
        # AuraDB Free has 200k nodes / 400k edges limit.
        # Elliptic has ~203k nodes. We might need to slice it slightly.
        MAX_NODES = 190000 
        if len(df) > MAX_NODES:
            logger.warning(f"Slicing dataset to {MAX_NODES} nodes to stay within AuraDB Free limits")
            df = df.iloc[:MAX_NODES]
            # Filter edges to only include those between existing nodes
            valid_tx_ids = set(df["txId"])
            edges_df = edges_df[edges_df["txId1"].isin(valid_tx_ids) & edges_df["txId2"].isin(valid_tx_ids)]
        
        adapter = Neo4jAdapter()
        
        # 3. Create Constraints (Idempotent)
        logger.info("Creating constraints in Neo4j")
        adapter.execute_query("CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE")
        
        # 4. Ingest Nodes
        logger.info(f"Ingesting {len(df)} nodes...")
        node_records = prepare_node_data(df)
        adapter.batch_save_nodes(node_records)
        
        # 5. Ingest Edges
        logger.info(f"Ingesting {len(edges_df)} edges...")
        edge_records = edges_df.to_dict('records')
        adapter.batch_save_edges(edge_records)
        
        logger.info("Neo4j population completed successfully")
        
    except Exception as e:
        logger.critical("Failed to populate Neo4j", extra={"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
