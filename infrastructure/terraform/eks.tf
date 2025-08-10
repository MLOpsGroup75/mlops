# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.name
  cluster_version = local.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # Disable extended support (cluster creator admin permissions)
  enable_cluster_creator_admin_permissions = false

  # Disable additional admin permissions
  create_cluster_primary_security_group_tags = false
  create_node_security_group                 = false

  # Disable extended support features
  enable_irsa                    = false
  create_cloudwatch_log_group    = false
  cluster_enabled_log_types      = []

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    for name, config in var.node_groups : name => {
      name           = "${local.name}-${name}"
      instance_types = config.instance_types
      capacity_type  = config.capacity_type

      min_size     = config.min_size
      max_size     = config.max_size
      desired_size = config.desired_size

      # Use the module's default launch template configuration
      # create_launch_template and launch_template_name removed to use defaults

      # Security group for additional access
      vpc_security_group_ids = [aws_security_group.additional.id]

      # IAM role policies for EBS CSI driver
      iam_role_additional_policies = {
        AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
      }
    }
  }

  # aws-auth configmap - temporarily disabled for initial deployment
  # Will be configured manually after cluster creation
  manage_aws_auth_configmap = false

  # aws_auth_roles = [
  #   {
  #     rolearn  = aws_iam_role.eks_admin.arn
  #     username = "eks-admin"
  #     groups   = ["system:masters"]
  #   },
  # ]

  # aws_auth_users = [
  #   {
  #     userarn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
  #     username = "cluster-creator"
  #     groups   = ["system:masters"]
  #   },
  # ]

  # Note: enable_cluster_creator_admin_permissions is only available in v20+
  # For v19, admin access is managed through aws_auth_configmap below

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.tags
}

# AWS Load Balancer Controller
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.6.2"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "false"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  depends_on = [
    module.eks,
    kubernetes_service_account.aws_load_balancer_controller,
  ]
}

# Service Account for AWS Load Balancer Controller
resource "kubernetes_service_account" "aws_load_balancer_controller" {
  metadata {
    name      = "aws-load-balancer-controller"
    namespace = "kube-system"
    labels = {
      "app.kubernetes.io/component" = "controller"
      "app.kubernetes.io/name"      = "aws-load-balancer-controller"
    }
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.aws_load_balancer_controller.arn
    }
  }

  depends_on = [module.eks]
}

# Metrics Server
resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  repository = "https://kubernetes-sigs.github.io/metrics-server/"
  chart      = "metrics-server"
  namespace  = "kube-system"
  version    = "3.11.0"

  depends_on = [module.eks]
}

# ArgoCD Installation
resource "helm_release" "argocd" {
  count = var.enable_argocd ? 1 : 0

  name             = "argocd"
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-cd"
  namespace        = "argocd"
  create_namespace = true
  version          = var.argocd_chart_version

  values = [
    file("${path.module}/helm-values/argocd-values.yaml")
  ]

  depends_on = [module.eks]
}

# Prometheus Stack (optional)
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0

  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  version          = var.prometheus_chart_version

  # Increase timeouts for complex chart installation
  timeout          = 900  # 15 minutes
  wait             = true
  wait_for_jobs    = true

  values = [
    file("${path.module}/helm-values/prometheus-minimal.yaml")
  ]

  depends_on = [module.eks]
}

# EBS CSI Driver Service Account and IAM Role
resource "aws_iam_role" "ebs_csi_driver" {
  name = "${local.name}-ebs-csi-driver"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = aws_iam_role.ebs_csi_driver.name
}

resource "kubernetes_service_account" "ebs_csi_controller" {
  metadata {
    name      = "ebs-csi-controller-sa"
    namespace = "kube-system"
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.ebs_csi_driver.arn
    }
  }

  depends_on = [module.eks]
}