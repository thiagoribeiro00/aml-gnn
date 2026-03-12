#!/bin/bash
# Script to deploy AML-GNN to Google Cloud Run

# Variables (Replace with your own or set via ENV)
PROJECT_ID=${GCP_PROJECT_ID:-"santodigital-ai-labs"}
REGION=${GCP_REGION:-"us-central1"}
IMAGE_NAME="aml-gnn-api"
SERVICE_NAME="aml-gnn-inference"
AR_REPO="aml-repository"

echo "🚀 Starting Deployment to Google Cloud Platform..."

# 1. Enable Services
echo "Checking GCP services..."
gcloud services enable artifactregistry.googleapis.com run.googleapis.com

# 2. Create Artifact Registry repository if not exists
echo "Setting up Artifact Registry..."
gcloud artifacts repositories create $AR_REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for AML GNN models" || true

# 3. Build and Push 
FULL_IMAGE_NAME="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$IMAGE_NAME:latest"
echo "Building Docker Image: $FULL_IMAGE_NAME"

docker build -t $FULL_IMAGE_NAME -f deploy/Dockerfile .

echo "Pushing Image to GCP..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
docker push $FULL_IMAGE_NAME

# 4. Deploy to Cloud Run
echo "Deploying to Cloud Run: $SERVICE_NAME"
gcloud run deploy $SERVICE_NAME \
    --image $FULL_IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars="ENVIRONMENT=prod,LOG_LEVEL=INFO" \
    --memory 2Gi \
    --cpu 2

echo "✅ Deployment successful!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format='value(status.url)')"
