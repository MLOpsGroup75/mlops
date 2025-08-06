#!/bin/bash

echo "🔧 Fixing ALB Access Issues..."
echo "=================================="

# Navigate to terraform directory
cd infrastructure/terraform

echo "1. Checking AWS authentication..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS authentication required"
    echo "Please run: aws configure"
    echo "Or if using SSO: aws sso login --profile YOUR_PROFILE"
    exit 1
fi

echo "✅ AWS authenticated"

echo "2. Applying updated IAM policy (removing restrictive conditions)..."
terraform apply -target=aws_iam_policy.aws_load_balancer_controller -auto-approve

if [ $? -eq 0 ]; then
    echo "✅ IAM policy updated"
else
    echo "❌ IAM policy update failed"
    exit 1
fi

echo "3. Getting ALB security group ID..."
ALB_SG_ID=$(terraform output -raw alb_security_group_id 2>/dev/null)

if [ -z "$ALB_SG_ID" ]; then
    echo "⚠️  ALB security group not found, creating output..."
    terraform apply -target=aws_security_group.alb -auto-approve
    ALB_SG_ID=$(terraform output -raw alb_security_group_id)
fi

echo "ALB Security Group ID: $ALB_SG_ID"

echo "4. Updating ingress to use ALB security group..."
cd ../..
kubectl annotate ingress mlops-ingress -n mlops-housing \
    alb.ingress.kubernetes.io/security-groups="$ALB_SG_ID" --overwrite

echo "5. Restarting ALB controller to pick up new permissions..."
kubectl delete pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller

echo "6. Waiting for ALB reconfiguration..."
sleep 60

echo "7. Testing ALB access..."
ALB_URL=$(kubectl get ingress mlops-ingress -n mlops-housing -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
if [ ! -z "$ALB_URL" ]; then
    echo "Testing: http://$ALB_URL/health"
    curl -s -m 10 -w "Status: %{http_code}\n" "http://$ALB_URL/health" || echo "Connection issue - may need more time"
else
    echo "ALB URL not found yet"
fi

echo ""
echo "🎉 ALB access fix completed!"
echo "=================================="
echo "Your services should be accessible at:"
echo "• API: http://$ALB_URL/api"
echo "• Predict: http://$ALB_URL/predict"  
echo "• Health: http://$ALB_URL/health"
echo ""
echo "If still not accessible, wait 2-3 minutes for full propagation."