#!/bin/bash

# AWS ECS Deployment Script for ShelfAware
# Replace YOUR_ACCOUNT_ID with your actual AWS account ID
# Replace ap-southeast-1 with your preferred region

set -e

ACCOUNT_ID="YOUR_ACCOUNT_ID"
REGION="ap-southeast-1"
BACKEND_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/shelfaware-backend"
FRONTEND_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/shelfaware-frontend"

echo "🚀 Starting ShelfAware ECS Deployment..."

# Authenticate Docker with ECR
echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push backend
echo "📦 Building backend image..."
docker build -t shelfaware-backend .

echo "🏷️  Tagging backend image..."
docker tag shelfaware-backend:latest $BACKEND_REPO:latest

echo "⬆️  Pushing backend image to ECR..."
docker push $BACKEND_REPO:latest

# Build and push frontend
echo "📦 Building frontend image..."
cd ref_frontend
docker build -t shelfaware-frontend .

echo "🏷️  Tagging frontend image..."
docker tag shelfaware-frontend:latest $FRONTEND_REPO:latest

echo "⬆️  Pushing frontend image to ECR..."
docker push $FRONTEND_REPO:latest

cd ..

echo "✅ Images pushed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Create ECS Cluster: aws ecs create-cluster --cluster-name shelfaware-cluster"
echo "2. Create Task Definitions (see task-definitions/ folder)"
echo "3. Create Services with Load Balancer"
echo "4. Update CORS in production with your ALB domain"
echo ""
echo "🔗 Backend ECR: $BACKEND_REPO"
echo "🔗 Frontend ECR: $FRONTEND_REPO"