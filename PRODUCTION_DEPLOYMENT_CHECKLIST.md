# Production Deployment Checklist for ShelfAware

## Pre-Deployment Preparation

### ✅ Environment Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Set `OPENAI_API_KEY` with production key
- [ ] Configure `DATABASE_URL` for PostgreSQL (RDS)
- [ ] Set Cognito credentials (region, user pool ID, client ID)
- [ ] Update CORS origins in `app/main.py` for production domains

### ✅ AWS Resources Setup
- [ ] Create ECR repositories: `shelfaware-backend` and `shelfaware-frontend`
- [ ] Set up RDS PostgreSQL database
- [ ] Create ECS cluster: `shelfaware-cluster`
- [ ] Configure VPC, subnets, and security groups
- [ ] Set up Application Load Balancer (ALB)
- [ ] Create target groups for backend (port 8000) and frontend (port 80)
- [ ] Configure Cognito User Pool and Client
- [ ] Set up CloudWatch log groups

### ✅ Code Preparation
- [ ] Update task definitions with actual AWS account ID and region
- [ ] Verify Dockerfiles build successfully
- [ ] Test application locally with production environment variables
- [ ] Run database migrations: `alembic upgrade head`
- [ ] Update frontend `.env` with production backend URL

## Deployment Execution

### ✅ Build and Push Images
- [ ] Authenticate Docker with ECR
- [ ] Build backend image: `docker build -t shelfaware-backend .`
- [ ] Tag and push backend image to ECR
- [ ] Build frontend image: `docker build -t shelfaware-frontend .` (from ref_frontend/)
- [ ] Tag and push frontend image to ECR

### ✅ ECS Setup
- [ ] Register backend task definition
- [ ] Register frontend task definition
- [ ] Create backend ECS service with ALB integration
- [ ] Create frontend ECS service with ALB integration
- [ ] Configure health checks for both services

### ✅ Database Setup
- [ ] Run initial database migrations in production
- [ ] Seed initial data if required
- [ ] Verify database connectivity from ECS tasks

## Post-Deployment Verification

### ✅ Application Health
- [ ] Backend health check: `GET /docs` or `GET /health`
- [ ] Frontend accessible via ALB domain
- [ ] API endpoints responding correctly
- [ ] Authentication flow working (login/register)

### ✅ Feature Verification
- [ ] User registration and login
- [ ] Book search and management
- [ ] Bookshelf creation and management
- [ ] Synopsis generation (test `/admin/sync-synopses`)
- [ ] CORS allowing frontend requests

### ✅ Monitoring Setup
- [ ] CloudWatch logs configured and accessible
- [ ] ALB access logs enabled
- [ ] Health check alarms configured
- [ ] Application performance monitoring

## Security & Compliance

### ✅ Security Hardening
- [ ] HTTPS enabled on ALB with ACM certificate
- [ ] Security groups configured (least privilege)
- [ ] IAM roles with minimal required permissions
- [ ] Secrets stored in AWS Secrets Manager (not env vars)
- [ ] Database encrypted at rest and in transit

### ✅ Backup & Recovery
- [ ] RDS automated backups configured
- [ ] ECS service auto-scaling configured
- [ ] Multi-AZ deployment for high availability
- [ ] Disaster recovery plan documented

## Performance Optimization

### ✅ Scaling Configuration
- [ ] ECS service auto-scaling based on CPU/memory
- [ ] RDS instance sized appropriately
- [ ] ALB configured for high traffic
- [ ] CDN setup for frontend assets (optional)

### ✅ Cost Optimization
- [ ] Monitor AWS costs and set up billing alerts
- [ ] Use appropriate instance types (Fargate Spot for dev)
- [ ] Configure log retention policies
- [ ] Set up resource cleanup (unused ECR images)

## Documentation & Maintenance

### ✅ Documentation
- [ ] Update README.md with production URLs
- [ ] Document environment variables and their purposes
- [ ] Create runbooks for common maintenance tasks
- [ ] Document backup and restore procedures

### ✅ Maintenance Tasks
- [ ] Set up automated dependency updates
- [ ] Configure monitoring alerts for key metrics
- [ ] Plan regular security updates and patches
- [ ] Establish incident response procedures

## Go-Live Checklist

- [ ] All health checks passing
- [ ] End-to-end user flows tested
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Backup verified
- [ ] Rollback plan ready
- [ ] Team notified of deployment
- [ ] Monitoring dashboards configured

## Emergency Contacts & Support

- AWS Support Plan: [Plan Level]
- Development Team: [Contact Info]
- On-call Rotation: [Schedule]
- Incident Response: [Process]

---

**Deployment Date:** __________
**Deployed By:** __________
**Version:** __________
**Notes:** __________

## Rollback Plan

If deployment fails:
1. Stop new ECS services
2. Roll back to previous task definition version
3. Update ALB to point to previous target groups
4. Verify application functionality
5. Investigate root cause before retrying