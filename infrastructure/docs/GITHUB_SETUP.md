# GitHub Repository Setup for MLOps AWS Deployment

This guide explains how to configure your GitHub repository with the necessary secrets and variables for automated deployment to AWS.

## 🔐 Required GitHub Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

### 1. Docker Hub Secrets
Click **"New repository secret"** and add these:

```
Secret Name: DOCKER_USERNAME
Secret Value: your-dockerhub-username

Secret Name: DOCKER_PASSWORD  
Secret Value: your-dockerhub-access-token

Secret Name: DOCKER_REPOSITORY
Secret Value: your-dockerhub-username/your-repository-name
```

**Example:**
```
DOCKER_USERNAME: johndoe
DOCKER_PASSWORD: dckr_pat_abc123xyz789...
DOCKER_REPOSITORY: johndoe/mlops-housing
```

### 2. AWS Access Secrets
```
Secret Name: AWS_ACCESS_KEY_ID
Secret Value: AKIA...

Secret Name: AWS_SECRET_ACCESS_KEY
Secret Value: your-secret-access-key
```

### 3. ArgoCD Secret (Optional)
```
Secret Name: ARGOCD_PASSWORD
Secret Value: your-argocd-admin-password
```

## 📝 Required GitHub Variables

Go to **Variables** tab and add these:

```
Variable Name: AWS_REGION
Variable Value: ap-south-1

Variable Name: EKS_CLUSTER_NAME
Variable Value: mlops-housing-dev
```

## 🐳 Docker Hub Setup

### 1. Create Docker Hub Repository
1. Go to [Docker Hub](https://hub.docker.com/)
2. Sign in to your account
3. Click **"Create Repository"**
4. Repository name: `mlops-housing` (or your preferred name)
5. Visibility: **Public** or **Private**
6. Click **"Create"**

### 2. Create Access Token
1. Go to **Account Settings** → **Security** → **Access Tokens**
2. Click **"New Access Token"**
3. Description: `GitHub Actions MLOps`
4. Permissions: **Read, Write, Delete**
5. Click **"Generate"**
6. **⚠️ Copy the token** - you won't see it again!

## 🔑 AWS IAM Setup

### 1. Deploy Infrastructure with Terraform
The IAM user and policies are now managed by Terraform for better consistency and security.

```bash
# Navigate to terraform directory
cd infrastructure/terraform

# Plan the changes
terraform plan

# Apply the changes
terraform apply
```

### 2. Create Access Keys (After Terraform Deployment)
Use the provided script to create access keys:

```bash
# Run the setup script
./scripts/setup_github_actions_iam.sh
```

This script will:
- Verify the IAM user exists
- Create new access keys
- Display the credentials to add to GitHub Secrets
- Test the S3 access

### 3. Manual IAM Setup (Alternative)
If you prefer to create the IAM user manually:

```bash
# Create IAM user
aws iam create-user --user-name github-actions-mlops

# Create access key
aws iam create-access-key --user-name github-actions-mlops
```

**Note**: The manual approach requires you to also create and attach the IAM policy manually. The Terraform approach is recommended.

## 📂 Repository File Updates

### 1. Update Docker Image References

**File: `infrastructure/kubernetes/api-service/deployment.yaml`**
```yaml
# Line ~30: Update image reference
image: your-dockerhub-username/your-repo/api-service:latest
```

**File: `infrastructure/kubernetes/predict-service/deployment.yaml`**
```yaml
# Line ~30: Update image reference  
image: your-dockerhub-username/your-repo/predict-service:latest
```

### 2. Update ArgoCD Application

**File: `infrastructure/kubernetes/argocd/application.yaml`**
```yaml
# Lines 14-15: Update repository URL
source:
  repoURL: https://github.com/your-username/your-repo.git
  
# Lines 46-47: Update allowed repositories
sourceRepos:
- 'https://github.com/your-username/your-repo.git'
```

### 3. Update Terraform Variables

**File: `infrastructure/terraform/terraform.tfvars`**
```hcl
# Copy from terraform.tfvars.example and update:
api_service_image     = "your-dockerhub-username/your-repo/api-service"
predict_service_image = "your-dockerhub-username/your-repo/predict-service"
```

## ✅ Verification Checklist

### GitHub Secrets ✓
- [ ] `DOCKER_USERNAME` set
- [ ] `DOCKER_PASSWORD` set (access token, not password)
- [ ] `DOCKER_REPOSITORY` set
- [ ] `AWS_ACCESS_KEY_ID` set
- [ ] `AWS_SECRET_ACCESS_KEY` set

### GitHub Variables ✓
- [ ] `AWS_REGION` set
- [ ] `EKS_CLUSTER_NAME` set

### Docker Hub ✓
- [ ] Repository created
- [ ] Access token generated with Read/Write permissions
- [ ] Token saved as `DOCKER_PASSWORD` secret

### AWS IAM ✓
- [ ] IAM user created for GitHub Actions
- [ ] Access keys generated
- [ ] Policy attached with EKS permissions

### File Updates ✓
- [ ] Kubernetes deployment images updated
- [ ] ArgoCD application repo URL updated
- [ ] Terraform variables configured

## 🧪 Test the Setup

### 1. Test CI/CD Pipeline
```bash
# Make a small change and push to main branch
echo "# Test" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push origin main

# Check GitHub Actions
# Go to GitHub repo → Actions tab → Watch the workflow
```

### 2. Verify Docker Images
```bash
# Check if images are pushed to Docker Hub
docker pull your-dockerhub-username/your-repo/api-service:latest
docker pull your-dockerhub-username/your-repo/predict-service:latest
```

### 3. Test AWS Access
```bash
# Test AWS CLI with the credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
aws sts get-caller-identity
```

## 🚨 Troubleshooting

### Common Issues

**❌ "authentication required" Docker error**
- Check `DOCKER_USERNAME` and `DOCKER_PASSWORD` are correct
- Ensure `DOCKER_PASSWORD` is an access token, not your password
- Verify repository exists and is accessible

**❌ "AWS credentials not found"**
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set
- Check IAM user has necessary permissions
- Ensure no trailing spaces in secret values

**❌ "AccessDenied" or "Forbidden" S3 errors**
- Verify IAM user has S3 permissions for the specific bucket
- Check if the S3 bucket exists and is accessible
- Ensure the IAM policy includes `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` permissions
- Run the setup script: `./scripts/setup_github_actions_iam.sh`

**❌ "repository does not exist" ArgoCD error**
- Update `argocd/application.yaml` with correct repository URL
- Ensure repository is public or ArgoCD has access
- Check the repository path is correct

**❌ "image pull error" in Kubernetes**
- Verify Docker Hub repository exists
- Check image names match in deployment files
- Ensure images were successfully pushed

### Debug Commands

```bash
# Check GitHub Actions logs
# Go to: GitHub repo → Actions → Click on failed workflow

# Check secrets are set
# Go to: GitHub repo → Settings → Secrets and variables → Actions

# Test Docker Hub access
docker login
docker push your-repo/test-image:latest

# Test AWS access
aws sts get-caller-identity
aws eks list-clusters --region ap-south-1
```

## 🔒 Security Best Practices

### GitHub Secrets
- ✅ Use access tokens, not passwords
- ✅ Set minimal required permissions
- ✅ Rotate credentials regularly
- ✅ Don't expose secrets in logs
- ❌ Never commit secrets to code

### AWS Security
- ✅ Use least privilege IAM policies
- ✅ Create dedicated IAM user for CI/CD
- ✅ Enable MFA for human users
- ✅ Monitor access with CloudTrail
- ❌ Don't use root account credentials

### Docker Hub
- ✅ Use private repositories for sensitive code
- ✅ Scan images for vulnerabilities
- ✅ Use specific image tags, not `latest` in production
- ✅ Implement image signing
- ❌ Don't store secrets in images

---

**Once configured, your repository will automatically build, test, and deploy your MLOps platform to AWS!** 🚀

Need help? Check the troubleshooting section or open a GitHub issue with details about your specific problem.