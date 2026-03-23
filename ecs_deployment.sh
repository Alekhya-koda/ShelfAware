# Build and push backend image
# 1. Authenticate Docker with ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# 2. Build backend image
docker build -t shelfaware-backend .

# 3. Tag for ECR
docker tag shelfaware-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/shelfaware-backend:latest

# 4. Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/shelfaware-backend:latest

# Build and push frontend image
cd ref_frontend
docker build -t shelfaware-frontend .
docker tag shelfaware-frontend:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/shelfaware-frontend:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/shelfaware-frontend:latest
