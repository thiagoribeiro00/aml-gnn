import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.application.data import DataService
from src.adapters.graph import Neo4jAdapter

def main():
    print("Setting up Neo4j Database...")
    svc = DataService()
    df = svc.clean_and_merge(svc.ingest_raw_data())
    
    adapter = Neo4jAdapter()
    adapter.execute_query("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE")
    
    nodes = [{"tx_id": int(r["txId"]), "timestep": int(r["timestep"]), "class": int(r["class"])} for _, r in df.head(10000).iterrows()]
    adapter.batch_save_nodes(nodes)
    print("Database setup completed.")

if __name__ == "__main__":
    main()
