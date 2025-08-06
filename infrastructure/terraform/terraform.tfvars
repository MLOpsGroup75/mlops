# Copy this file to terraform.tfvars and update with your specific values
# cp terraform.tfvars.example terraform.tfvars

# Project Configuration
project_name = "mlops-housing"
environment  = "dev"  # dev, staging, prod
aws_region   = "ap-south-1"  # Mumbai region

# VPC Configuration (adjust if needed)
vpc_cidr        = "10.0.0.0/16"
private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

# EKS Configuration
cluster_version = "1.28"

# Node Groups Configuration (no auto-scaling)
node_groups = {
  main = {
    instance_types = ["t3.medium"]
    min_size       = 2
    max_size       = 3
    desired_size   = 2
    capacity_type  = "SPOT"
  }
}

# Application Configuration
docker_registry        = "docker.io"
api_service_image      = "sudhagar/api-service"
predict_service_image  = "sudhagar/predict-service"
image_tag              = "latest"

# Feature Flags
enable_argocd     = true
enable_monitoring = true

# Versions (update as needed)
argocd_chart_version     = "5.51.6"
prometheus_chart_version = "58.2.1"

# Domain Configuration (not required)
# Custom domain not needed as per specifications

# Examples for different environments:

# Simplified Configuration (current setup):
# - Fixed replicas (no auto-scaling)
# - Single node group
# - ap-south-1 region
# - No custom domain required

# For minimal development setup:
# node_groups = {
#   main = {
#     instance_types = ["t3.small"]
#     min_size       = 1
#     max_size       = 2
#     desired_size   = 1
#     capacity_type  = "ON_DEMAND"
#   }
# }