#!/bin/bash

# EKS Deployment Debug Script
# This script helps diagnose why the K8s service isn't getting deployed/restarted

echo "🔍 EKS Deployment Debug Script"
echo "=============================="
echo ""

# Configuration
AWS_REGION=${AWS_REGION:-"ap-south-1"}
EKS_CLUSTER_NAME=${EKS_CLUSTER_NAME:-"mlops-housing-dev"}
NAMESPACE="mlops-housing"
CURRENT_COMMIT=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short HEAD)

echo "📋 Configuration:"
echo "AWS Region: $AWS_REGION"
echo "EKS Cluster: $EKS_CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
echo "Current Commit: $CURRENT_COMMIT"
echo "Short Commit: $SHORT_COMMIT"
echo ""

# Update kubeconfig
echo "🔧 Step 1: Updating kubeconfig..."
aws eks update-kubeconfig --region $AWS_REGION --name $EKS_CLUSTER_NAME

echo ""
echo "📋 Step 2: Checking current deployment status..."
kubectl get deployment api-service -n $NAMESPACE -o wide
echo ""
echo "Current API service image:"
CURRENT_IMAGE=$(kubectl get deployment api-service -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "$CURRENT_IMAGE"

echo ""
echo "📋 Step 3: Checking deployment manifest image..."
MANIFEST_IMAGE=$(grep "image:" infrastructure/kubernetes/api-service/deployment.yaml | awk '{print $2}')
echo "Manifest image: $MANIFEST_IMAGE"

echo ""
echo "📋 Step 4: Checking pods and their images..."
kubectl get pods -n $NAMESPACE -o custom-columns="NAME:.metadata.name,IMAGE:.spec.containers[*].image,STATUS:.status.phase,AGE:.metadata.creationTimestamp"

echo ""
echo "📋 Step 5: Checking pod age and restart counts..."
kubectl get pods -n $NAMESPACE -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[*].restartCount,AGE:.metadata.creationTimestamp"

echo ""
echo "📋 Step 6: Checking deployment rollout status..."
kubectl rollout status deployment/api-service -n $NAMESPACE --timeout=10s

echo ""
echo "📋 Step 7: Checking deployment events..."
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' --field-selector involvedObject.name=api-service | tail -10

echo ""
echo "📋 Step 8: Checking ArgoCD application status..."
if kubectl get application mlops-housing-platform -n argocd >/dev/null 2>&1; then
    echo "ArgoCD Application found:"
    kubectl get application mlops-housing-platform -n argocd -o yaml | grep -A 10 "status:"
else
    echo "⚠️ ArgoCD application not found or not accessible"
fi

echo ""
echo "📋 Step 9: Expected vs Actual Image Comparison..."
EXPECTED_IMAGE="sudhagar/api-service:$CURRENT_COMMIT"
echo "Expected image: $EXPECTED_IMAGE"
echo "Current image:  $CURRENT_IMAGE"

if [[ "$CURRENT_IMAGE" == "$EXPECTED_IMAGE" ]]; then
    echo "✅ Images match!"
else
    echo "❌ Images don't match - deployment needs update"
fi

echo ""
echo "🛠️ Step 10: Suggested Fix Actions..."
echo ""

if [[ "$CURRENT_IMAGE" != "$EXPECTED_IMAGE" ]]; then
    echo "🔧 Option 1: Update deployment manifest and trigger ArgoCD sync"
    echo "   sed -i 's|image: .*/api-service:.*|image: sudhagar/api-service:$CURRENT_COMMIT|g' infrastructure/kubernetes/api-service/deployment.yaml"
    echo "   git add infrastructure/kubernetes/api-service/deployment.yaml"
    echo "   git commit -m 'Update API service image to $SHORT_COMMIT'"
    echo "   git push"
    echo ""
    
    echo "🔧 Option 2: Force deployment restart (temporary fix)"
    echo "   kubectl rollout restart deployment api-service -n $NAMESPACE"
    echo ""
    
    echo "🔧 Option 3: Update image directly and restart"
    echo "   kubectl set image deployment/api-service api-service=sudhagar/api-service:$CURRENT_COMMIT -n $NAMESPACE"
    echo ""
fi

echo "🔧 Option 4: Test the current API endpoints"
ALB_URL=$(kubectl get ingress mlops-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
if [ ! -z "$ALB_URL" ]; then
    echo "   Test health: curl http://$ALB_URL/health"
    echo "   Test API: curl -X POST http://$ALB_URL/api/v1/predict -H 'Content-Type: application/json' -d '{...}'"
else
    echo "   ALB URL not available yet"
fi

echo ""
echo "✅ Debug completed!"
echo ""
echo "🚨 Common Issues:"
echo "• CD pipeline didn't update the manifest file"
echo "• ArgoCD sync policy not triggering properly"
echo "• Image pull issues from Docker Hub"
echo "• Resource constraints preventing pod restart"
