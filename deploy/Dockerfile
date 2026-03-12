# Production-ready Dockerfile for AML-GNN
# Multi-stage build to reduce final image size

# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install all dependencies in Stage 1
# This ensures transitive dependencies like 'packaging' are resolved correctly
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir \
    torch \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Stage 2: Final Image
FROM python:3.10-slim

WORKDIR /app

# Copy the entire local environment from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
# Ensure python looks into the copied local packages
ENV PYTHONPATH="/root/.local/lib/python3.10/site-packages"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV ENVIRONMENT="prod"
ENV LOG_LEVEL="INFO"

# Default command
CMD ["python", "scripts/run_training.py", "--epochs", "50"]
