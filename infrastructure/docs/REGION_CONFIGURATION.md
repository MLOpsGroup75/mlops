# AWS Region Configuration: ap-south-1 (Mumbai)

## 📍 Region Overview

The MLOps platform is configured to deploy in **ap-south-1 (Asia Pacific - Mumbai)** region.

### Key Configuration Changes Made:

## 🔧 Terraform Configuration

### **1. Region Settings**
```hcl
# infrastructure/terraform/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"  # Changed from us-west-2
}

variable "azs" {
  description = "Availability zones"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b", "ap-south-1c"]
}
```

### **2. Node Group Configuration (No Auto-scaling)**
```hcl
# Simplified node group - no auto-scaling
node_groups = {
  main = {
    instance_types = ["t3.medium"]
    min_size       = 1         # Reduced from 2
    max_size       = 3         # Reduced from 10
    desired_size   = 2         # Reduced from 3
    capacity_type  = "ON_DEMAND"
  }
  # Removed spot instance node group
}
```

## 🎯 Application Configuration

### **1. Fixed Replicas (No Auto-scaling)**
```yaml
# API Service: 1 replica (reduced from 3)
# Predict Service: 1 replica (reduced from 2)

# infrastructure/kubernetes/api-service/deployment.yaml
spec:
  replicas: 1  # Fixed, no auto-scaling

# infrastructure/kubernetes/predict-service/deployment.yaml  
spec:
  replicas: 1  # Fixed, no auto-scaling
```

### **2. HPA Disabled**
```yaml
# infrastructure/kubernetes/hpa.yaml
# All HPA configurations commented out
# Auto-scaling disabled as per requirements
```

### **3. Pod Disruption Budget Adjusted**
```yaml
# infrastructure/kubernetes/hpa.yaml
spec:
  minAvailable: 1  # Adjusted for single replica setup
```

## 🌐 CI/CD Pipeline Updates

### **1. GitHub Actions**
```yaml
# .github/workflows/ci-docker.yml
aws-region: ${{ vars.AWS_REGION || 'ap-south-1' }}  # Changed default

# Update kubeconfig command
aws eks update-kubeconfig --region ap-south-1 --name mlops-housing-dev
```

### **2. Required GitHub Variables**
```
AWS_REGION: ap-south-1
EKS_CLUSTER_NAME: mlops-housing-dev
```

## 🚫 Removed Features

### **1. Custom Domain Support**
- Removed all custom domain configurations
- Simplified ingress to use ALB DNS only
- No SSL/TLS certificate setup needed

### **2. Auto-scaling Components**
- **HPA (Horizontal Pod Autoscaler)**: Completely disabled
- **Spot Instances**: Removed spot node group
- **Cluster Autoscaler**: Limited scaling range (1-3 nodes)

### **3. Advanced Networking**
- Simplified to basic ALB ingress
- No custom domain routing
- No SSL termination

## 💰 Cost Impact

### **Estimated Monthly Costs (ap-south-1)**
- **EKS Control Plane**: ~₹5,400 ($72)
- **Worker Nodes**: 2 t3.medium instances ~₹4,500-6,750 ($60-90)  
- **Load Balancer**: ~₹1,200 ($16)
- **EBS Storage**: ~₹750 ($10)
- **Total**: ~₹11,850-13,650 ($158-182) per month

### **Cost Optimizations Applied**
- ✅ **Fixed replicas**: No unexpected scaling costs
- ✅ **Single node group**: Simplified instance management
- ✅ **No spot instances**: Predictable costs (but higher than spot)
- ✅ **Minimal storage**: Basic persistent volume needs
- ✅ **No domain costs**: No Route53 or certificate fees

## 📋 Deployment Commands

### **1. Terraform Deployment**
```bash
cd infrastructure/terraform

# Configure for ap-south-1
terraform init
terraform plan -var="aws_region=ap-south-1" -var="project_name=mlops-housing"
terraform apply -var="aws_region=ap-south-1" -var="project_name=mlops-housing"
```

### **2. kubectl Configuration**
```bash
# Configure kubectl for ap-south-1
aws eks --region ap-south-1 update-kubeconfig --name mlops-housing-dev

# Verify cluster access
kubectl get nodes
```

### **3. Application Access**
```bash
# Get ALB URL
kubectl get ingress -n mlops-housing

# Test endpoints
ALB_URL=$(kubectl get ingress mlops-ingress -n mlops-housing -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$ALB_URL/api/health
curl http://$ALB_URL/predict/health
```

## 🔍 Verification Checklist

### **Infrastructure**
- [ ] EKS cluster deployed in ap-south-1
- [ ] Node group with 2 t3.medium instances
- [ ] VPC spans ap-south-1a, ap-south-1b, ap-south-1c
- [ ] ALB created and healthy

### **Applications**
- [ ] API service: 1 replica running
- [ ] Predict service: 1 replica running
- [ ] No HPA resources created
- [ ] Services accessible via ALB

### **Monitoring**
- [ ] Prometheus deployed (if enabled)
- [ ] Grafana accessible (if enabled)
- [ ] ArgoCD operational

## 🎯 Benefits of This Configuration

### **Simplicity**
- ✅ **Fixed scaling**: Predictable resource usage
- ✅ **Single region**: No cross-region complexity
- ✅ **No custom domains**: Simplified networking
- ✅ **Reduced components**: Easier troubleshooting

### **Cost Control**
- ✅ **Predictable costs**: No surprise scaling charges
- ✅ **Right-sized**: 1 replica each for development/testing
- ✅ **Regional pricing**: ap-south-1 competitive pricing
- ✅ **No premium features**: Basic load balancing only

### **Operational**
- ✅ **Low complexity**: Minimal moving parts
- ✅ **Regional latency**: Good for Indian users
- ✅ **Straightforward**: Easy to understand and maintain
- ✅ **Compliance**: Data remains in India region

## 🔄 Future Scaling Options

### **When You Need Auto-scaling**
1. **Uncomment HPA configurations** in `infrastructure/kubernetes/hpa.yaml`
2. **Adjust replicas** in deployment files
3. **Add spot instances** back to node groups
4. **Increase node group limits** in Terraform

### **When You Need Custom Domain**
1. **Purchase domain** and configure Route53
2. **Request SSL certificate** via ACM
3. **Uncomment ingress TLS** configurations
4. **Update ArgoCD and monitoring** ingress rules

### **When You Need Multi-region**
1. **Create terraform workspaces** for each region
2. **Deploy additional clusters** in us-east-1, eu-west-1, etc.
3. **Set up cross-region** replication and routing
4. **Configure global** load balancing

---

**This configuration provides a production-ready MLOps platform optimized for Indian deployment with controlled costs and simplified operations!** 🇮🇳🚀