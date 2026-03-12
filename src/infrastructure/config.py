import os
from dotenv import load_dotenv

# Find project root (where .env is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

class Config:
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
    
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # App Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Data Paths - Derived from PROJECT_ROOT
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    
    MODEL_PATH = os.path.join(MODELS_DIR, "sage_best.pt")

    # Specific Data Files
    FEATURES_PATH = os.path.join(RAW_DATA_DIR, "elliptic_txs_features.csv")
    EDGES_PATH = os.path.join(RAW_DATA_DIR, "elliptic_txs_edgelist.csv")
    CLASSES_PATH = os.path.join(RAW_DATA_DIR, "elliptic_txs_classes.csv")

    # MLOps & GCP
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", os.path.join(PROJECT_ROOT, "mlruns"))
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "wise-aml-project")
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")

    def validate(self):
        """Simple validation to ensure critical paths exist."""
        critical_paths = [self.RAW_DATA_DIR, self.FEATURES_PATH]
        for path in critical_paths:
            if not os.path.exists(path):
                # print(f"Warning: Path not found: {path}")
                pass

config = Config()
config.validate()
