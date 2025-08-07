# AWS EKS MLOps Platform Deployment Guide

This guide walks you through deploying your MLOps Housing Price Prediction platform on AWS using EKS, ArgoCD, and Terraform.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│   Docker Hub     │───▶│   AWS EKS       │
│                 │    │                  │    │                 │
│ • Source Code   │    │ • Container      │    │ • API Service   │
│ • Kubernetes    │    │   Images         │    │ • Predict Svc   │
│ • CI/CD         │    │                  │    │ • Monitoring    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
                                               ┌─────────────────┐
                                               │     ArgoCD      │
                                               │                 │
                                               │ • GitOps        │
                                               │ • Auto Deploy  │
                                               │ • Sync Policy   │
                                               └─────────────────┘
```

## 📋 Prerequisites

### 1. Required Tools
```bash
# Install required tools
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 2. AWS Configuration
```bash
# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region (us-west-2), and output format (json)

# Verify access
aws sts get-caller-identity
```

### 3. Docker Hub Setup
Ensure you have:
- Docker Hub account
- Repository created (e.g., `your-username/mlops-housing`)
- GitHub Secrets configured:
  - `DOCKER_USERNAME`
  - `DOCKER_PASSWORD`
  - `DOCKER_REPOSITORY`

## 🚀 Deployment Steps

### Step 1: Update Configuration

1. **Update Terraform Variables**
```bash
cd infrastructure/terraform
cp variables.tf variables.tf.backup

# Edit variables.tf with your specific values:
# - aws_region
# - project_name
# - docker_registry
# - api_service_image
# - predict_service_image
```

2. **Update Kubernetes Manifests**
```bash
# Update image references in:
# - infrastructure/kubernetes/api-service/deployment.yaml
# - infrastructure/kubernetes/predict-service/deployment.yaml
# Replace: your-dockerhub-username/your-repo/api-service:latest
# With:    YOUR_ACTUAL_DOCKER_REPO/api-service:latest
```

3. **Update ArgoCD Application**
```bash
# Edit infrastructure/kubernetes/argocd/application.yaml
# Replace: https://github.com/your-username/your-repo.git
# With:    YOUR_ACTUAL_GITHUB_REPO_URL
```

### Step 2: Deploy Infrastructure

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan the deployment
terraform plan -var="aws_region=ap-south-1" -var="project_name=mlops-housing"

# Apply the infrastructure
terraform apply -var="aws_region=ap-south-1" -var="project_name=mlops-housing"
```

**⏱️ Expected Time:** 15-20 minutes

### Step 3: Configure kubectl

```bash
# Update kubeconfig
aws eks --region ap-south-1 update-kubeconfig --name mlops-housing-dev

# Verify cluster access
kubectl get nodes
kubectl get namespaces
```

### Step 4: Access ArgoCD

```bash
# Get ArgoCD admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Port forward to ArgoCD server
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Access ArgoCD at: https://localhost:8080
# Username: admin
# Password: (from step above)
```

### Step 5: Deploy Applications via ArgoCD

```bash
# Apply ArgoCD application
kubectl apply -f infrastructure/kubernetes/argocd/application.yaml

# Create namespace and apply base manifests
kubectl apply -f infrastructure/kubernetes/namespaces/
kubectl apply -f infrastructure/kubernetes/api-service/
kubectl apply -f infrastructure/kubernetes/predict-service/
kubectl apply -f infrastructure/kubernetes/ingress.yaml
kubectl apply -f infrastructure/kubernetes/hpa.yaml
```

### Step 6: Access Applications

```bash
# Get ALB URL
kubectl get ingress -n mlops-housing

# Test endpoints
ALB_URL=$(kubectl get ingress mlops-ingress -n mlops-housing -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# API Health Check
curl http://$ALB_URL/api/health

# Predict Service Health Check  
curl http://$ALB_URL/predict/health

# Make a prediction
curl -X POST http://$ALB_URL/predict/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "median_house_value": 452600.0,
    "ocean_proximity": "NEAR BAY"
  }'
```

### Step 7: Access Monitoring

```bash
# Access Grafana
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80

# Access at: http://localhost:3000
# Username: admin
# Password: admin123!

# Access Prometheus
kubectl port-forward svc/prometheus-prometheus -n monitoring 9090:9090
# Access at: http://localhost:9090
```

## 🔄 CI/CD Integration

### Automatic Deployments

Your CI/CD pipeline is configured to:

1. **Build & Push**: Docker images to Docker Hub
2. **Security Scan**: Vulnerability scanning with Trivy
3. **Integration Test**: Health checks and functionality tests
4. **Image Update**: ArgoCD automatically detects new images
5. **Deploy**: Rolling updates to EKS cluster

### Manual Deployment Trigger

```bash
# Force ArgoCD sync
kubectl patch app mlops-housing-platform -n argocd -p '{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}' --type merge

# Or via ArgoCD UI:
# 1. Go to ArgoCD UI
# 2. Click on mlops-housing-platform
# 3. Click "Sync" button
```

## 📊 Monitoring & Observability

### Application Metrics
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **ServiceMonitors**: Automatic service discovery

### Key Dashboards
1. **Kubernetes Cluster Overview**
2. **Application Performance**
3. **API Service Metrics**
4. **Predict Service Metrics**
5. **Resource Utilization**

### Alerts
- High error rates
- Resource exhaustion
- Pod failures
- Network issues

## 🔧 Troubleshooting

### Common Issues

1. **Pods not starting**
```bash
kubectl describe pod <pod-name> -n mlops-housing
kubectl logs <pod-name> -n mlops-housing
```

2. **Ingress not working**
```bash
kubectl describe ingress mlops-ingress -n mlops-housing
kubectl get events -n mlops-housing
```

3. **ArgoCD sync issues**
```bash
kubectl logs -n argocd deployment/argocd-application-controller
kubectl get app mlops-housing-platform -n argocd -o yaml
```

### Debug Commands
```bash
# Check cluster status
kubectl get nodes
kubectl get pods -A

# Check specific namespace
kubectl get all -n mlops-housing

# Check logs
kubectl logs -f deployment/api-service -n mlops-housing
kubectl logs -f deployment/predict-service -n mlops-housing

# Check ArgoCD
kubectl get apps -n argocd
kubectl describe app mlops-housing-platform -n argocd
```

## 🧹 Cleanup

### Destroy Infrastructure
```bash
# Delete Kubernetes resources
kubectl delete -f infrastructure/kubernetes/

# Destroy Terraform infrastructure
cd infrastructure/terraform
terraform destroy
```

**⚠️ Warning:** This will delete all resources and data!

## 💡 Production Considerations

### Security
- [ ] Enable SSL/TLS certificates
- [ ] Configure proper RBAC
- [ ] Set up VPN access
- [ ] Enable audit logging
- [ ] Configure security groups

### Scalability
- [ ] Configure cluster autoscaler
- [ ] Set up multiple AZs
- [ ] Configure persistent storage
- [ ] Implement caching layers
- [ ] Set up CDN

### Reliability
- [ ] Configure backups
- [ ] Set up disaster recovery
- [ ] Implement health checks
- [ ] Configure alerting
- [ ] Document runbooks

### Cost Optimization
- [ ] Use Spot instances
- [ ] Configure resource limits
- [ ] Implement auto-scaling
- [ ] Monitor costs
- [ ] Use reserved instances

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Kubernetes events
3. Check ArgoCD application status
4. Review application logs
5. Consult AWS EKS documentation

**Next Steps:** Once deployed, you can extend the platform with additional ML models, data pipelines, and monitoring capabilities!