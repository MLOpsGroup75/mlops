# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "alb_security_group_id" {
  description = "ALB Security Group ID"
  value       = aws_security_group.alb.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

# EKS Cluster Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = module.eks.cluster_version
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

# EKS Node Group Outputs
output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
}

output "eks_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS managed node groups"
  value       = module.eks.eks_managed_node_groups_autoscaling_group_names
}

# Region
output "region" {
  description = "AWS region"
  value       = var.aws_region
}

# ArgoCD
output "argocd_server_url" {
  description = "ArgoCD server URL (when using LoadBalancer)"
  value       = var.enable_argocd ? "kubectl port-forward svc/argocd-server -n argocd 8080:443" : "ArgoCD not enabled"
}

# Monitoring
output "grafana_url" {
  description = "Grafana URL (when using LoadBalancer)"
  value       = var.enable_monitoring ? "kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80" : "Monitoring not enabled"
}

# Kubectl Configuration
output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ap-south-1 update-kubeconfig --name ${module.eks.cluster_name}"
}

# Application URLs (when deployed)
output "api_service_url" {
  description = "API Service URL (after ArgoCD deployment)"
  value       = "http://<ALB_URL>/api"
}

output "predict_service_url" {
  description = "Predict Service URL (after ArgoCD deployment)"
  value       = "http://<ALB_URL>/predict"
}

# Next Steps
output "next_steps" {
  description = "Next steps after infrastructure deployment"
  value = <<EOF
1. Configure kubectl: ${module.eks.cluster_name}
   aws eks --region ap-south-1 update-kubeconfig --name ${module.eks.cluster_name}

2. Verify cluster access:
   kubectl get nodes

3. Access ArgoCD (if enabled):
   kubectl port-forward svc/argocd-server -n argocd 8080:443
   # Username: admin
   # Password: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

4. Access Grafana (if enabled):
   kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
   # Username: admin  
   # Password: prom-operator

5. Deploy your applications:
   kubectl apply -f ../kubernetes/

6. Check application status:
   kubectl get pods -n mlops-housing
EOF
}