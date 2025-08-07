# Docker Hub Integration Setup

This document explains how to configure your GitHub repository to push Docker images to Docker Hub instead of GitHub Container Registry.

## Required GitHub Secrets

You need to add the following secrets to your GitHub repository:

### 1. Go to GitHub Repository Settings
- Navigate to your repository on GitHub
- Click on **Settings** → **Secrets and variables** → **Actions**

### 2. Add Required Secrets

Click **New repository secret** and add these three secrets:

#### `DOCKER_USERNAME`
- **Value**: Your Docker Hub username
- **Example**: `johnsmith` or `mycompany`

#### `DOCKER_PASSWORD`
- **Value**: Your Docker Hub access token (recommended) or password
- **⚠️ Important**: Use an access token instead of your password for security
- **How to create access token**:
  1. Go to [Docker Hub](https://hub.docker.com/)
  2. Login → Account Settings → Security → Access Tokens
  3. Click **New Access Token**
  4. Give it a name like "GitHub Actions"
  5. Copy the generated token

#### `DOCKER_REPOSITORY`
- **Value**: Your Docker Hub repository name
- **Format**: `username/repository-name`
- **Example**: `johnsmith/mlops-housing` or `mycompany/mlops-services`

## Docker Hub Repository Setup

### 1. Create Repository on Docker Hub
- Go to [Docker Hub](https://hub.docker.com/)
- Click **Create Repository**
- Repository name should match what you put in `DOCKER_REPOSITORY`
- Choose **Public** or **Private** as needed

### 2. Repository Naming Convention
With the current setup, your images will be tagged as:
- `your-repo/api-service:latest`
- `your-repo/api-service:main`
- `your-repo/api-service:sha-abc123`
- `your-repo/predict-service:latest`
- `your-repo/predict-service:main`
- `your-repo/predict-service:sha-abc123`

## Workflow Changes Made

✅ **Updated Environment Variables**:
```yaml
env:
  REGISTRY: docker.io
  DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
  DOCKER_REPOSITORY: ${{ secrets.DOCKER_REPOSITORY }}
```

✅ **Updated Login Actions**:
```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ env.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_PASSWORD }}
```

✅ **Updated Image Names**:
```yaml
images: ${{ env.DOCKER_REPOSITORY }}/api-service
images: ${{ env.DOCKER_REPOSITORY }}/predict-service
```

✅ **Updated Security Scanning**:
```yaml
image-ref: ${{ env.DOCKER_REPOSITORY }}/api-service:${{ github.sha }}
image-ref: ${{ env.DOCKER_REPOSITORY }}/predict-service:${{ github.sha }}
```

## Testing the Setup

1. **Add the three secrets** to your GitHub repository
2. **Push a commit** to `main` or `develop` branch
3. **Check Actions tab** to see the workflow running
4. **Verify on Docker Hub** that images are being pushed

## Example Configuration

If your Docker Hub username is `mycompany` and you want to call your repository `mlops-housing`, set:

```
DOCKER_USERNAME: mycompany
DOCKER_PASSWORD: dckr_pat_abc123... (your access token)
DOCKER_REPOSITORY: mycompany/mlops-housing
```

This will create images like:
- `mycompany/mlops-housing/api-service:latest`
- `mycompany/mlops-housing/predict-service:latest`

## Benefits of Docker Hub

✅ **Public Registry**: Free public repositories  
✅ **Better Performance**: Often faster than GitHub registry  
✅ **Standard**: Most widely used Docker registry  
✅ **Integration**: Works with all Docker tools  
✅ **Pull Limits**: 200 pulls per 6 hours for free accounts  

## Security Notes

- ⚠️ **Never use your Docker Hub password directly**
- ✅ **Always use access tokens** for GitHub Actions
- 🔒 **Rotate tokens periodically**
- 👀 **Monitor your Docker Hub repository** for unexpected activity