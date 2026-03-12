# AML-GNN: Anti-Money Laundering with Graph Neural Networks

A robust, production-ready system for detecting illicit financial transactions using Graph Neural Networks (GNNs), built on **PyTorch Geometric** and following **Clean Architecture** principles.

---

## 🚀 Technical Overview

This project implements a node classification approach on transaction graphs to identify money laundering patterns. Unlike traditional machine learning that treats transactions as independent events, GNNs capture the relational context—essential for detecting complex laundering schemes like layering and smurfing.

### 🧠 Algorithm selection & Rationale

We implement two state-of-the-art GNN architectures:

1.  **GraphSAGE (Sample and Aggregate)**:
    *   **Why**: Designed for large-scale graphs, GraphSAGE doesn't require the entire graph during training. It learns an inductive mapping that generalizes to unseen nodes by sampling and aggregating features from a node's local neighborhood.
    *   **Implementation**: A 2-layer `SAGEConv` architecture. The first layer aggregates neighbor features, and the second refines the representation for binary classification (Licit vs. Illicit).

2.  **GAT (Graph Attention Network)**:
    *   **Why**: In financial networks, not all neighbors are equally important. GAT uses multi-head attention mechanisms to assign different weights to neighbors, allowing the model to focus on high-risk transaction paths.
    *   **Implementation**: Utilizes `GATConv` with 8 attention heads. It applies `ELU` activation and dropout for regularization.

---

## 🏗️ Architecture

The project strictly follows **Clean Architecture** to ensure maintainability and testability:

-   **Domain**: Core logic and value objects (`src/domain`).
-   **Use Cases**: Orchestrates data flow (Ingestion, Feature Engineering, Training, Inference).
-   **Adapters**: bridges the gap between the domain and external tools:
    *   **PytorchAdapter**: Converts Neo4j/Dataframe data into `torch_geometric.data.Data` objects.
    *   **LoggerAdapter**: Structured logging and metrics tracking via **MLflow**.
-   **Infrastructure**: Configurations, storage handlers, and database drivers (Neo4j).

---

## 🛠️ Data Pipeline & Implementation

1.  **Ingestion**: Data is fetched from **Neo4j AuraDB**, representing the transaction network where nodes are transactions and edges are the flow of funds.
2.  **Feature Engineering**:
    *   **Scaling**: 165 features (from the Elliptic dataset) are standardized using `StandardScaler`.
    *   **Temporal Split**: To prevent data leakage, we use a time-based split:
        *   **Train**: Timesteps 1-34
        *   **Validation**: Timesteps 35-42
        *   **Test**: Timesteps 43-49
3.  **Training**: Managed by `GNNTrainer`, supporting weighted loss functions to handle the inherent class imbalance in AML data (Licit >> Illicit).
4.  **Observability**: Integrated with **MLflow** for experiment tracking and model versioning.

---

## 🚦 How to Run

### 1. Prerequisites
- Python 3.9+
- Neo4j AuraDB instance (configured in `.env`)
- Docker (optional)

### 2. Setup
```bash
make setup
```

### 3. Data Ingestion
Populate your Neo4j database with the initial dataset:
```bash
make ingest
```

### 4. Training
Run the training pipeline (defaults to GraphSAGE):
```bash
# To train GraphSAGE
make train model=sage

# To train GAT
make train model=gat
```

### 5. Serving
Start the FastAPI inference service:
```bash
make api
```

### 6. Interface
Run the Streamlit interactive dashboard to visualize predictions:
```bash
make gui
```

---

## 📁 Project Structure

```text
├── data/               # Local data storage (raw, interim, processed)
├── deploy/             # Docker and cloud deployment configs
├── models/             # Saved model checkpoints (.pt)
├── scripts/            # Entry point scripts for the pipeline
├── src/
│   ├── adapters/       # PyTorch, Neo4j, and Logger adapters
│   ├── domain/         # Business logic & entities
│   ├── infrastructure/ # DB/Config/Storage implementation
│   ├── interfaces/     # API (FastAPI) & Frontend (Streamlit)
│   ├── models/         # GNN Architectures (GraphSAGE, GAT)
│   └── use_cases/      # Modular pipeline steps
└── tests/              # Unit and integration tests
```
