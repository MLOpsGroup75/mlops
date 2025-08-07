# Docker Hub Setup Verification

## Quick Checklist ✅

After setting up Docker Hub integration, verify these items:

### 1. Docker Hub Repository
- [ ] Repository exists on Docker Hub
- [ ] Repository name matches `DOCKER_REPOSITORY` secret
- [ ] Repository has correct visibility (Public/Private)

### 2. GitHub Secrets
- [ ] `DOCKER_USERNAME` = Your Docker Hub username
- [ ] `DOCKER_PASSWORD` = Your Docker Hub access token (NOT password)
- [ ] `DOCKER_REPOSITORY` = `username/repository-name`

### 3. Access Token Permissions
- [ ] Token has "Read, Write" permissions minimum
- [ ] Token is not expired
- [ ] Token was copied correctly (no spaces/newlines)

### 4. Repository Format
Your images will be pushed as:
```
your-username/your-repo-name/api-service:main
your-username/your-repo-name/predict-service:main
```

## Test the Setup

1. **Push a commit** to `main` or `develop` branch
2. **Go to Actions tab** in GitHub
3. **Check the workflow logs** for:
   ```
   Successfully pushed your-repo/api-service:main
   Successfully pushed your-repo/predict-service:main
   ```
4. **Verify on Docker Hub** that images appear in your repository

## Common Issues

### ❌ "repository does not exist"
- Check that `DOCKER_REPOSITORY` matches your actual Docker Hub repo
- Ensure the repository exists on Docker Hub

### ❌ "insufficient_scope: authorization failed"  
- Use an **access token**, not your password
- Ensure token has **Read, Write** permissions
- Check that `DOCKER_USERNAME` is correct

### ❌ "authentication required"
- Verify all 3 secrets are set in GitHub
- Check for typos in secret names (`DOCKER_USERNAME`, `DOCKER_PASSWORD`, `DOCKER_REPOSITORY`)

### ❌ "denied: requested access to the resource is denied"
- Repository might be private and token lacks permissions
- Repository name in secret might be wrong

### ❌ "MANIFEST_UNKNOWN: manifest unknown; unknown tag=<sha>"
- **FIXED**: Changed SHA format to match security scanning requirements
- This happens when there's a mismatch between generated SHA tags and expected SHA format
- **Root cause**: Security scanning uses full 40-char SHA (`${{ github.sha }}`) but Docker metadata action generates short 7-char SHA by default
- **Solution**: Use `type=sha,format=long` to generate full SHA tags

**Current configuration (fixed):**
```yaml
tags: |
  type=ref,event=branch
  type=ref,event=pr
  type=sha,format=long        # Generates full 40-char SHA tags
  type=raw,value=latest,enable={{is_default_branch}}
```

**This ensures SHA tags match between:**
- ✅ Docker image tags: `your-repo/api-service:f3b34f89dd57d4b7d1a2429298e8aba9fb55139c`
- ✅ Security scanning: `sha-f3b34f89dd57d4b7d1a2429298e8aba9fb55139c` format

### ❌ "Resource not accessible by integration" (Security Scanning)
- **FIXED**: Added `security-events: write` permission to security-scan job
- **FIXED**: Split SARIF uploads into separate steps with distinct categories
- This happens when GitHub Actions lacks permission to upload security scan results
- **Solution**: Ensure security-scan job has proper permissions and separate upload steps

**Current security scanning configuration (fixed):**
```yaml
security-scan:
  permissions:
    security-events: write
    actions: read
    contents: read
  steps:
    - name: Upload API service Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'api-service-trivy-results.sarif'
        category: 'api-service'
```

## Success Indicators

✅ **GitHub Actions logs show:**
```
Successfully authenticated to Docker Hub
Building docker image...
Pushing docker image...
Successfully pushed image
```

✅ **Docker Hub shows:**
- New images in your repository
- Recent push timestamps
- Correct tag names (main, latest, sha-abc123)