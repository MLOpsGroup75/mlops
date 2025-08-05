# Project Configuration
variable "project_name" {
  description = "Name of the MLOps project"
  type        = string
  default     = "mlops-housing"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "azs" {
  description = "Availability zones"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b", "ap-south-1c"]
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS Configuration
variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "EKS node group configuration"
  type = map(object({
    instance_types = list(string)
    min_size       = number
    max_size       = number
    desired_size   = number
    capacity_type  = string
  }))
  default = {
    main = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3
      desired_size   = 2
      capacity_type  = "SPOT"
    }
  }
}

# ArgoCD Configuration
variable "argocd_chart_version" {
  description = "ArgoCD Helm chart version"
  type        = string
  default     = "5.51.6"
}

variable "enable_argocd" {
  description = "Enable ArgoCD installation"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable Prometheus and Grafana monitoring stack"
  type        = bool
  default     = true
}

variable "prometheus_chart_version" {
  description = "Prometheus Helm chart version"
  type        = string
  default     = "25.8.0"
}

# Application Configuration
variable "docker_registry" {
  description = "Docker registry for application images"
  type        = string
  default     = "docker.io"
}

variable "api_service_image" {
  description = "API service Docker image"
  type        = string
  default     = "your-dockerhub-username/your-repo/api-service"
}

variable "predict_service_image" {
  description = "Predict service Docker image"
  type        = string
  default     = "your-dockerhub-username/your-repo/predict-service"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

variable "enable_ssl" {
  description = "Enable SSL/TLS with AWS Certificate Manager"
  type        = bool
  default     = false
}