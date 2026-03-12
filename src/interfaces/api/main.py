from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import torch
import os
import sys
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.adapters.logger_adapter import StructuredLogger
from src.models.gnn_architecture import GraphSAGEModel
from src.use_cases.predict_and_save import InferencePipeline
from src.infrastructure.storage import storage

# Initialize Logger
StructuredLogger.configure(level=config.LOG_LEVEL, environment=config.ENVIRONMENT)
logger = StructuredLogger.get_logger("inference_api")

app = FastAPI(title="AML GNN Inference Service", version="1.0.0")

# Prometheus Metrics
PREDICTION_COUNTER = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of prediction requests")

# Global variables for model and pipeline
model = None
pipeline = None

class PredictionRequest(BaseModel):
    node_id: int
    features: List[float]
    neighbors: List[int] # For real-time graph reconstruction if needed

class PredictionResponse(BaseModel):
    tx_id: int
    risk_score: float
    is_illicit: bool
    processing_time_ms: float

@app.on_event("startup")
async def load_model():
    global model, pipeline
    try:
        logger.info("Loading model for inference", extra={"path": config.MODEL_PATH})
        # For Wise-scale, we'd load the "Production" model from MLflow Registry
        # But for now, we load from the local path
        if not os.path.exists(config.MODEL_PATH):
            logger.error("Model file not found", extra={"path": config.MODEL_PATH})
            return

        # Initialize architecture (assuming sage for now)
        # Note: In production, we'd read metadata to know which architecture to use
        model = GraphSAGEModel(in_channels=165, hidden_channels=64, out_channels=2)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        pipeline = InferencePipeline(model=model, device="cpu")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical("Failed to load model on startup", extra={"error": str(e)})

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.perf_counter()
    PREDICTION_COUNTER.inc()
    
    with PREDICTION_LATENCY.time():
        try:
            # For a real GNN prediction, we need the node + its neighborhood
            # In MVP, we might simulate or expect pre-processed features
            # Here we just show the structure
            
            # Simulated x and edge_index for this specific node
            x = torch.tensor([request.features], dtype=torch.float)
            edge_index = torch.tensor([[], []], dtype=torch.long) # Simplified for single node
            
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index)
            
            results = pipeline.predict(data)
            risk_score = float(results.iloc[0]["risk_score"])
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info("Inference completed", extra={
                "node_id": request.node_id,
                "risk_score": risk_score,
                "duration_ms": duration_ms
            })
            
            return PredictionResponse(
                tx_id=request.node_id,
                risk_score=risk_score,
                is_illicit=risk_score > 0.5,
                processing_time_ms=duration_ms
            )
        except Exception as e:
            logger.error("Prediction failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail="Inference error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
