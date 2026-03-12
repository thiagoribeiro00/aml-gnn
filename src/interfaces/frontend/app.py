import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import torch
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.infrastructure.storage import storage
from src.models.gnn_architecture import GraphSAGEModel
from src.use_cases.predict_and_save import InferencePipeline

# --- Page Config ---
st.set_page_config(
    page_title="AML-GNN Monitor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AML-GNN Transaction Monitoring Dashboard")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose Mode", ["Risk Dashboard", "Inference Simulator", "Model Metrics"])

# --- Load Data Cache ---
@st.cache_data
def load_processed_data():
    status = st.empty()
    try:
        status.info("📂 Loading processed data...")
        data_path = os.path.join(config.PROCESSED_DATA_DIR, "scaled_data.pkl")
        
        if not os.path.exists(data_path):
            status.warning(f"⚠️ File not found: {data_path}")
            return None
            
        df = storage.load_pickle(data_path)
        
        preds_path = os.path.join(config.DATA_DIR, "results", "predictions.csv")
        if os.path.exists(preds_path):
            status.info("📂 Integrating risk scores...")
            preds = pd.read_csv(preds_path)
            result = pd.concat([df.reset_index(drop=True), preds["risk_score"]], axis=1)
            status.success("✅ Data loaded successfully!")
            time.sleep(1) # Visual feedback
            status.empty()
            return result
        
        status.warning("⚠️ Risk scores not found. Run 'make predict'.")
        time.sleep(1)
        status.empty()
        return df
    except Exception as e:
        status.error(f"❌ Fatal error loading data: {e}")
        return None

# --- Main Logic ---
data = load_processed_data()

if app_mode == "Risk Dashboard":
    st.header("📊 Transaction Risk Analysis")
    
    if data is not None and "risk_score" in data.columns:
        col1, col2, col3 = st.columns(3)
        
        total_txs = len(data)
        high_risk = len(data[data["risk_score"] > 0.5])
        avg_risk = data["risk_score"].mean()
        
        col1.metric("Total Transactions", f"{total_txs:,}")
        col2.metric("High Risk (>0.5)", high_risk)
        col3.metric("Average Risk", f"{avg_risk:.2%}")
        
        st.markdown("### Risk Score Distribution")
        fig = px.histogram(data, x="risk_score", nbins=50, 
                           title="Scores Distribution",
                           labels={'risk_score': 'Risk Score'},
                           color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Suspicious Transactions (Top 50)")
        top_risky = data.sort_values(by="risk_score", ascending=False).head(50)
        st.dataframe(top_risky[["txId", "risk_score", "timestep"]], use_container_width=True)
    else:
        st.warning("Please run 'make predict' first to generate risk scores.")

elif app_mode == "Inference Simulator":
    st.header("🧪 GNN Inference Simulator")
    
    st.markdown("""
    This simulator allows you to test the loaded model with manual data.
    *(Simplified interface for demonstration)*
    """)
    
    with st.form("inference_form"):
        tx_id = st.number_input("Transaction ID", value=12345)
        # For simplicity, we just use a random seed for missing features or mean
        st.info("The GNN model requires 165 features. Provide the main ones:")
        f1 = st.slider("Feature 1 (Local context)", -2.0, 2.0, 0.0)
        f2 = st.slider("Feature 2 (Transaction amount scaled)", -2.0, 2.0, 0.0)
        
        submitted = st.form_submit_button("Analyze Transaction")
        
        if submitted:
            with st.spinner("Processing..."):
                # Load model
                if os.path.exists(config.MODEL_PATH):
                    model = GraphSAGEModel(in_channels=165, hidden_channels=64, out_channels=2)
                    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
                    model.eval()
                    
                    # Mock data for single node
                    x = torch.zeros((1, 165))
                    x[0, 0] = f1
                    x[0, 1] = f2
                    
                    pipeline = InferencePipeline(model=model, device="cpu")
                    from torch_geometric.data import Data
                    single_data = Data(x=x, edge_index=torch.tensor([[], []], dtype=torch.long))
                    
                    res = pipeline.predict(single_data)
                    score = float(res.iloc[0]["risk_score"])
                    
                    if score > 0.5:
                        st.error(f"⚠️ ALERT: High Risk detected! Score: {score:.4f}")
                    else:
                        st.success(f"✅ Safe Transaction. Score: {score:.4f}")
                else:
                    st.error("Model not found at 'models/sage_best.pt'. Run 'make train'.")

elif app_mode == "Model Metrics":
    st.header("📈 Performance and MLOps")
    
    # Simple Metrics Visualization
    st.markdown("### Training Evolution")
    # In a real scenario, we would pull this from MLflow API
    # Here we show a placeholder interactive chart
    df_metrics = pd.DataFrame({
        'Epoch': range(1, 11),
        'Loss': [0.5, 0.4, 0.35, 0.28, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08],
        'F1-Score': [0.65, 0.70, 0.75, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.93]
    })
    
    fig_metric = px.line(df_metrics, x='Epoch', y=['Loss', 'F1-Score'], 
                         title="Training Progress (Demo)",
                         markers=True)
    st.plotly_chart(fig_metric, use_container_width=True)
    
    st.info("For detailed metrics and complete versioning, access the MLflow dashboard.")

st.sidebar.markdown("---")
st.sidebar.caption("Developed for Wise AML-GNN Integration")
