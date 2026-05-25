import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.infrastructure.config import config
from src.domain.models import ModelFactory
from src.application.gnn import GNNService
from src.application.graphrag import GraphRAGService
from src.adapters.llm import GeminiAdapter
from src.adapters.graph import Neo4jAdapter

st.set_page_config(page_title="AML-GNN Senior Dashboard", layout="wide")

@st.cache_resource
def load_gnn(mtype: str):
    path = os.path.join(config.MODELS_DIR, f"{mtype}_best.pt")
    if not os.path.exists(path): return None
    return ModelFactory.load(mtype, path, in_channels=165)

@st.cache_resource
def load_rag():
    return GraphRAGService(GeminiAdapter(), Neo4jAdapter())

st.title("🏦 AML-GNN Transaction Monitoring")

import re

if prompt := st.chat_input("Ex: Por que a transação 230490356 é de risco?"):
    with st.spinner("Generating explanation..."):
        try:
            # Senior logic: Extract numeric ID from text
            ids = re.findall(r'\d+', prompt)
            if not ids:
                st.warning("Please include a Transaction ID in your question.")
                st.stop()
            
            target_id = ids[0]
            rag = load_rag()
            explanation = rag.explain_prediction(target_id, 0.99)
            st.chat_message("user").write(prompt)
            st.chat_message("assistant").write(explanation)
        except Exception as e:
            st.error(f"Error: {e}")
