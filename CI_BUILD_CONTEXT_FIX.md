# CI/CD Build Context Fix

## 🐛 **Problem**
GitHub Actions CI/CD was failing with:
```
ERROR: failed to build: failed to solve: failed to compute cache key: 
failed to calculate checksum of ref: "/services/predict/app": not found
```

## 🔍 **Root Cause**
**Build Context Mismatch Between Local Docker Compose and CI/CD**

After fixing `docker-compose.yml` to use root build context (`.`), the GitHub Actions workflow was still using the old individual service directory contexts (`./services/api`, `./services/predict`).

### What Was Happening:
1. **Dockerfiles updated** to expect files from root context:
   ```dockerfile
   COPY services/api/requirements.txt .
   COPY services/common/ services/common/
   COPY config/ config/
   ```

2. **CI workflow still using old context**:
   ```bash
   docker build -t api-service:test ./services/api  # ❌ Wrong context
   ```

3. **Build failure** because shared directories not accessible

## ✅ **Solution Applied**

### 1. **Updated Integration Test Builds**
```bash
# BEFORE (broken):
docker build -t api-service:test ./services/api
docker build -t predict-service:test ./services/predict

# AFTER (fixed):
docker build -t api-service:test -f ./services/api/Dockerfile .
docker build -t predict-service:test -f ./services/predict/Dockerfile .
```

### 2. **Updated Docker Build-Push Actions**
```yaml
# BEFORE (broken):
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    context: ./services/api
    file: ./services/api/Dockerfile

# AFTER (fixed):
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: ./services/api/Dockerfile
```

### 3. **Applied to Both Services**
- ✅ API service build context updated
- ✅ Predict service build context updated
- ✅ Integration test builds updated
- ✅ Docker build-push actions updated

## 🎯 **Results**

✅ **Local Testing Confirmed**: Both services build successfully with new commands
✅ **Consistent Context**: CI/CD now matches docker-compose approach
✅ **Shared Dependencies**: All builds can access `services/common/` and `config/`
✅ **Cache Efficiency**: GitHub Actions can properly cache layers

## 🚀 **Verification Commands**

```bash
# These commands now work in CI:
docker build -t api-service:test -f ./services/api/Dockerfile .
docker build -t predict-service:test -f ./services/predict/Dockerfile .

# And these work locally:
docker compose build
docker compose up -d
```

## 📋 **CI/CD Workflow Changes Summary**

| Component | Before | After |
|-----------|--------|-------|
| Integration test builds | `./services/api` | `. -f ./services/api/Dockerfile` |
| API build-push context | `./services/api` | `.` |
| Predict build-push context | `./services/predict` | `.` |
| Dockerfile paths | `Dockerfile` | `./services/api/Dockerfile` |

## 🔧 **Key Principle**

**Consistency is critical**: When you change build context in one place (docker-compose.yml), you must update ALL build processes (CI/CD, local builds, etc.) to use the same context.

**Root Context Benefits**:
- ✅ Access to shared code across services
- ✅ Consistent build behavior everywhere
- ✅ Simplified dependency management
- ✅ Better Docker layer caching

**The CI/CD pipeline now perfectly matches the local Docker Compose setup!** 🎉