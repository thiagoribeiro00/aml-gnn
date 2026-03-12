from neo4j import GraphDatabase
from .config import config
from src.adapters.logger_adapter import StructuredLogger, log_execution_time

logger = StructuredLogger.get_logger(__name__)

class Neo4jClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
            cls._instance._driver = None
        return cls._instance

    @log_execution_time(logger)
    def connect(self):
        if not self._driver:
            try:
                self._driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
                )
                self._driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j database")
            except Exception as e:
                logger.error("Failed to connect to Neo4j database", extra={"error": str(e)})
                raise

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j database connection closed")
            self._driver = None

    @property
    def driver(self):
        if not self._driver:
            self.connect()
        return self._driver

neo4j_client = Neo4jClient()
