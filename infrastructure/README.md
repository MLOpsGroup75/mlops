# MLOps AWS Infrastructure

Complete Infrastructure as Code (IaC) solution for deploying MLOps Housing Price Prediction platform on AWS using EKS, ArgoCD, and Terraform.

## 🚀 Quick Start

### Prerequisites
- AWS CLI configured
- Terraform >= 1.0
- kubectl
- Helm 3
- Docker Hub account with repository

### 1. Deploy Infrastructure
```bash
cd infrastructure/terraform

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your specific values

# Deploy
terraform init
terraform plan
terraform apply
```

### 2. Configure Access
```bash
# Update kubeconfig
aws eks --region ap-south-1 update-kubeconfig --name mlops-housing-dev

# Verify access
kubectl get nodes
```

### 3. Deploy Applications
```bash
# Apply Kubernetes manifests
kubectl apply -f ../kubernetes/namespaces/
kubectl apply -f ../kubernetes/api-service/
kubectl apply -f ../kubernetes/predict-service/
kubectl apply -f ../kubernetes/ingress.yaml
kubectl apply -f ../kubernetes/hpa.yaml

# Setup ArgoCD for GitOps
kubectl apply -f ../kubernetes/argocd/
```

### 4. Access Services
```bash
# Get ALB URL
kubectl get ingress -n mlops-housing

# Access ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Access Grafana
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
```

## 📁 Directory Structure

```
infrastructure/
├── terraform/                 # AWS Infrastructure (IaC)
│   ├── main.tf                # Main Terraform configuration
│   ├── variables.tf           # Input variables
│   ├── outputs.tf             # Output values
│   ├── vpc.tf                 # VPC and networking
│   ├── eks.tf                 # EKS cluster configuration
│   ├── iam.tf                 # IAM roles and policies
│   ├── terraform.tfvars.example # Example variables file
│   └── helm-values/           # Helm chart customizations
│       ├── argocd-values.yaml # ArgoCD configuration
│       └── prometheus-values.yaml # Monitoring stack config
├── kubernetes/                # Kubernetes manifests
│   ├── namespaces/            # Namespace definitions
│   ├── api-service/           # API service deployment
│   ├── predict-service/       # Predict service deployment
│   ├── argocd/                # ArgoCD application definitions
│   ├── ingress.yaml           # ALB ingress configuration
│   └── hpa.yaml               # Auto-scaling configuration
└── docs/                      # Documentation
    ├── DEPLOYMENT_GUIDE.md    # Step-by-step deployment
    └── ARCHITECTURE.md        # Detailed architecture docs
```

## 🏗️ Architecture Overview

### Core Components
- **EKS Cluster**: Managed Kubernetes with auto-scaling
- **ArgoCD**: GitOps continuous deployment
- **ALB**: Application Load Balancer for external access
- **Prometheus + Grafana**: Monitoring and observability
- **Auto-scaling**: HPA for pods, CA for nodes

### Services
- **API Service**: FastAPI REST API (1 replica, fixed)
- **Predict Service**: ML inference service (1 replica, fixed)
- **Monitoring**: Prometheus metrics collection
- **Visualization**: Grafana dashboards

### Networking
- **VPC**: 10.0.0.0/16 with public/private subnets
- **Load Balancer**: Internet-facing ALB
- **Service Discovery**: Kubernetes DNS
- **Security**: Security groups and network policies

## 🔄 GitOps Workflow

1. **Code Push** → GitHub repository
2. **CI/CD Pipeline** → GitHub Actions
3. **Container Build** → Docker Hub
4. **Manifest Update** → Automatic image tag update
5. **ArgoCD Sync** → Deployment to EKS
6. **Monitoring** → Prometheus + Grafana

## 🛠️ Configuration

### Required GitHub Secrets
```
# Docker Hub
DOCKER_USERNAME
DOCKER_PASSWORD
DOCKER_REPOSITORY

# AWS Access (for deployment)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

### Required GitHub Variables
```
AWS_REGION=ap-south-1
EKS_CLUSTER_NAME=mlops-housing-dev
```

### Update Image References
Before deployment, update the following files with your Docker Hub repository:
- `infrastructure/kubernetes/api-service/deployment.yaml`
- `infrastructure/kubernetes/predict-service/deployment.yaml`
- `infrastructure/kubernetes/argocd/application.yaml`

## 📊 Monitoring & Access

### ArgoCD UI
```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Access: https://localhost:8080
# Username: admin
# Password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### Grafana UI
```bash
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
# Access: http://localhost:3000
# Username: admin
# Password: admin123!
```

### Application Endpoints
```bash
# Get ALB URL
ALB_URL=$(kubectl get ingress mlops-ingress -n mlops-housing -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# API Health
curl http://$ALB_URL/api/health

# Predict Health
curl http://$ALB_URL/predict/health

# Make Prediction
curl -X POST http://$ALB_URL/predict/predict \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88, ...}'
```

## 🔧 Troubleshooting

### Common Commands
```bash
# Check cluster status
kubectl get nodes
kubectl get pods -A

# Check application pods
kubectl get pods -n mlops-housing
kubectl describe pod <pod-name> -n mlops-housing
kubectl logs <pod-name> -n mlops-housing

# Check ArgoCD
kubectl get app -n argocd
kubectl describe app mlops-housing-platform -n argocd

# Check ingress
kubectl get ingress -n mlops-housing
kubectl describe ingress mlops-ingress -n mlops-housing
```

### Debug Issues
1. **Pods not starting**: Check resource limits and image availability
2. **Ingress not working**: Verify ALB controller and security groups
3. **ArgoCD sync issues**: Check repository access and image tags
4. **Resource issues**: Check HPA and cluster autoscaler

## 💰 Cost Considerations

### Default Configuration (ap-south-1)
- **EKS Control Plane**: ~₹5,400/month ($72)
- **Worker Nodes**: 2 t3.medium instances (~₹4,500-6,750/month)
- **Load Balancer**: ~₹1,200/month ($16)
- **EBS Storage**: ~₹750/month ($10)
- **Total**: ~₹11,850-13,650/month ($158-182)

### Cost Optimization
- Fixed replicas prevent unexpected scaling costs
- Single node group simplifies management
- Monitor with AWS Cost Explorer
- Implement resource quotas

## 🧹 Cleanup

```bash
# Delete Kubernetes resources
kubectl delete -f infrastructure/kubernetes/

# Destroy Terraform infrastructure
cd infrastructure/terraform
terraform destroy
```

## 📚 Documentation

- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Detailed step-by-step instructions
- **[Architecture](docs/ARCHITECTURE.md)**: Complete technical architecture
- **[Terraform Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)**
- **[ArgoCD Docs](https://argo-cd.readthedocs.io/)**
- **[EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes to infrastructure or manifests
4. Test with `terraform plan`
5. Submit pull request

## 📞 Support

For issues:
1. Check troubleshooting section
2. Review logs: `kubectl logs -f deployment/<service> -n mlops-housing`
3. Check ArgoCD UI for sync status
4. Verify AWS resources in console
5. Open GitHub issue with details

---

**Ready to deploy your MLOps platform on AWS with production-ready infrastructure!** 🚀