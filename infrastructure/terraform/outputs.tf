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

# S3 Outputs - Datasets Bucket
output "s3_datasets_bucket_name" {
  description = "Name of the S3 bucket for MLOps datasets"
  value       = aws_s3_bucket.mlops_housing_datasets.bucket
}

output "s3_datasets_bucket_arn" {
  description = "ARN of the S3 bucket for MLOps datasets"
  value       = aws_s3_bucket.mlops_housing_datasets.arn
}

output "s3_datasets_bucket_regional_domain_name" {
  description = "Regional domain name of the S3 datasets bucket"
  value       = aws_s3_bucket.mlops_housing_datasets.bucket_regional_domain_name
}

# S3 Outputs - Artifacts Bucket
output "s3_artifacts_bucket_name" {
  description = "Name of the S3 bucket for MLOps model artifacts"
  value       = aws_s3_bucket.mlops_housing_artifacts.bucket
}

output "s3_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for MLOps model artifacts"
  value       = aws_s3_bucket.mlops_housing_artifacts.arn
}

output "s3_artifacts_bucket_regional_domain_name" {
  description = "Regional domain name of the S3 artifacts bucket"
  value       = aws_s3_bucket.mlops_housing_artifacts.bucket_regional_domain_name
}

# IAM Role Output
output "s3_access_role_arn" {
  description = "ARN of the IAM role for S3 access from EKS"
  value       = aws_iam_role.s3_access_role.arn
}

# GitHub Actions IAM User Output
output "github_actions_user_arn" {
  description = "ARN of the GitHub Actions IAM user"
  value       = aws_iam_user.github_actions_mlops.arn
}

output "github_actions_policy_arn" {
  description = "ARN of the GitHub Actions IAM policy"
  value       = aws_iam_policy.github_actions_mlops.arn
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

7. S3 Bucket Information:
   
   Datasets Bucket:
   - Name: ${aws_s3_bucket.mlops_housing_datasets.bucket}
   - ARN: ${aws_s3_bucket.mlops_housing_datasets.arn}
   - Upload datasets: aws s3 cp your-dataset.csv s3://${aws_s3_bucket.mlops_housing_datasets.bucket}/datasets/
   
   Model Artifacts Bucket:
   - Name: ${aws_s3_bucket.mlops_housing_artifacts.bucket}
   - ARN: ${aws_s3_bucket.mlops_housing_artifacts.arn}
   - Upload models: aws s3 cp model.pkl s3://${aws_s3_bucket.mlops_housing_artifacts.bucket}/models/
   - Upload experiments: aws s3 cp experiment-results.json s3://${aws_s3_bucket.mlops_housing_artifacts.bucket}/experiments/
   
   Access from pods using the service account: s3-access-sa in mlops-housing namespace
EOF
}