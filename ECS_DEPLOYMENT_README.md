# AWS ECS Deployment Guide for ShelfAware

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed locally
4. **ECR Repositories** created:
   - `shelfaware-backend`
   - `shelfaware-frontend`
5. **VPC and Subnets** configured
6. **Application Load Balancer** (ALB) created
7. **RDS PostgreSQL** database (for production)
8. **Cognito User Pool** configured

## Step 1: Update Configuration Files

### 1.1 Update Account ID and Region
Edit `deploy-to-ecs.sh`:
```bash
ACCOUNT_ID="YOUR_ACTUAL_ACCOUNT_ID"
REGION="ap-southeast-1"  # or your preferred region
```

### 1.2 Update Task Definitions
Edit both task definition files in `task-definitions/`:
- Replace `YOUR_ACCOUNT_ID` with your AWS account ID
- Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key
- Update database URL with your RDS endpoint
- Update Cognito configuration with your actual values

## Step 2: Create ECR Repositories

```bash
# Create ECR repositories
aws ecr create-repository --repository-name shelfaware-backend --region ap-southeast-1
aws ecr create-repository --repository-name shelfaware-frontend --region ap-southeast-1
```

## Step 3: Deploy Images

```bash
# Make script executable and run
chmod +x deploy-to-ecs.sh
./deploy-to-ecs.sh
```

## Step 4: Create ECS Resources

### 4.1 Create Cluster
```bash
aws ecs create-cluster --cluster-name shelfaware-cluster --region ap-southeast-1
```

### 4.2 Register Task Definitions
```bash
# Backend
aws ecs register-task-definition --cli-input-json file://task-definitions/backend-task-definition.json --region ap-southeast-1

# Frontend
aws ecs register-task-definition --cli-input-json file://task-definitions/frontend-task-definition.json --region ap-southeast-1
```

### 4.3 Create Services

First, create target groups for your ALB:

```bash
# Backend target group (port 8000)
aws elbv2 create-target-group \
  --name shelfaware-backend-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id YOUR_VPC_ID \
  --target-type ip \
  --region ap-southeast-1

# Frontend target group (port 80)
aws elbv2 create-target-group \
  --name shelfaware-frontend-tg \
  --protocol HTTP \
  --port 80 \
  --vpc-id YOUR_VPC_ID \
  --target-type ip \
  --region ap-southeast-1
```

Create services:

```bash
# Backend service
aws ecs create-service \
  --cluster shelfaware-cluster \
  --service-name shelfaware-backend-service \
  --task-definition shelfaware-backend \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abc123],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:region:account:targetgroup/backend-tg/123456,containerName=shelfaware-backend,containerPort=8000" \
  --region ap-southeast-1

# Frontend service
aws ecs create-service \
  --cluster shelfaware-cluster \
  --service-name shelfaware-frontend-service \
  --task-definition shelfaware-frontend \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abc123],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:region:account:targetgroup/frontend-tg/123456,containerName=shelfaware-frontend,containerPort=80" \
  --region ap-southeast-1
```

## Step 5: Update CORS Configuration

Once your ALB domains are available, update the CORS configuration in `app/main.py`:

```python
# Replace the placeholder with your actual ALB domain
CORS_ORIGINS = [
    "http://your-alb-domain.com",
    "https://your-alb-domain.com"
]
```

## Step 6: Database Migration

Run database migrations in production:

```bash
# Connect to your ECS backend container and run
alembic upgrade head
```

## Step 7: Health Checks

Configure health check endpoints in your ALB target groups:
- Backend: `/health` or `/docs`
- Frontend: `/` (serves index.html)

## Monitoring

### CloudWatch Logs
Both services are configured to send logs to CloudWatch. Check:
- `/ecs/shelfaware-backend`
- `/ecs/shelfaware-frontend`

### Application Monitoring
- Monitor ECS service health
- Set up CloudWatch alarms for CPU/memory usage
- Configure X-Ray for distributed tracing (optional)

## Troubleshooting

### Common Issues:
1. **Image Pull Errors**: Ensure ECR repositories exist and IAM permissions are correct
2. **Task Failures**: Check CloudWatch logs for application errors
3. **Database Connection**: Verify RDS security groups and connection string
4. **CORS Issues**: Update origins after ALB domain is available

### Useful Commands:
```bash
# Check service status
aws ecs describe-services --cluster shelfaware-cluster --services shelfaware-backend-service

# View task logs
aws logs tail /ecs/shelfaware-backend --region ap-southeast-1

# Update service
aws ecs update-service --cluster shelfaware-cluster --service shelfaware-backend-service --task-definition shelfaware-backend:NEW_REVISION
```

## Security Considerations

1. **Secrets Management**: Use AWS Secrets Manager for sensitive data
2. **IAM Roles**: Create specific roles with minimal required permissions
3. **Network Security**: Configure security groups properly
4. **SSL/TLS**: Enable HTTPS on ALB with ACM certificates

## Cost Optimization

- Use Fargate Spot for non-critical workloads
- Configure auto-scaling based on CPU/memory usage
- Set up billing alerts
- Monitor resource utilization