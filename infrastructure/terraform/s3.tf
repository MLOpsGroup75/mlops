# S3 Bucket for MLOps Dataset Storage
resource "aws_s3_bucket" "mlops_housing_datasets" {
  bucket = "${local.name}-datasets"

  tags = merge(local.tags, {
    Purpose = "MLOps dataset storage"
    DataType = "housing-datasets"
  })
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "mlops_housing_datasets" {
  count  = var.enable_s3_versioning ? 1 : 0
  bucket = aws_s3_bucket.mlops_housing_datasets.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Server-side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_housing_datasets" {
  bucket = aws_s3_bucket.mlops_housing_datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Public Access Block (Security Best Practice)
resource "aws_s3_bucket_public_access_block" "mlops_housing_datasets" {
  bucket = aws_s3_bucket.mlops_housing_datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket Lifecycle Configuration
resource "aws_s3_bucket_lifecycle_configuration" "mlops_housing_datasets" {
  count  = var.s3_lifecycle_enabled ? 1 : 0
  bucket = aws_s3_bucket.mlops_housing_datasets.id

  rule {
    id     = "dataset_lifecycle"
    status = "Enabled"

    filter {
      prefix = "datasets/"
    }

    # Move to Infrequent Access after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Move to Glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Move to Deep Archive after 365 days
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    # Clean up incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # Rule for versioned objects cleanup
  rule {
    id     = "version_cleanup"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 730  # Keep versions for 2 years
    }
  }
}

# IAM Policy for S3 Access
data "aws_iam_policy_document" "s3_access_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.mlops_housing_datasets.arn,
      "${aws_s3_bucket.mlops_housing_datasets.arn}/*"
    ]
  }
}

# IAM Role for EKS Pods to Access S3
resource "aws_iam_role" "s3_access_role" {
  name = "${local.name}-s3-access-role"

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
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:mlops-housing:s3-access-sa"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

# Attach the S3 access policy to the role
resource "aws_iam_role_policy" "s3_access_policy" {
  name   = "${local.name}-s3-access-policy"
  role   = aws_iam_role.s3_access_role.id
  policy = data.aws_iam_policy_document.s3_access_policy.json
}

# Service Account for S3 Access
resource "kubernetes_service_account" "s3_access" {
  metadata {
    name      = "s3-access-sa"
    namespace = "mlops-housing"
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.s3_access_role.arn
    }
  }

  depends_on = [module.eks]
}

# Kubernetes namespace for MLOps
resource "kubernetes_namespace" "mlops_housing" {
  metadata {
    name = "mlops-housing"
    labels = {
      project = var.project_name
      environment = var.environment
    }
  }

  depends_on = [module.eks]
}
