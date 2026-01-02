# EKS Deployment Guide
## Voice Quality Evaluator

---

## Prerequisites (from IT Team)

Before deploying, ensure you have:
- [ ] ECR Repository URI
- [ ] EKS cluster access (kubeconfig)
- [ ] Kubernetes namespace created
- [ ] IRSA service account configured
- [ ] AWS CLI and kubectl installed

---

## Step 1: Configure AWS & kubectl

```bash
# Login to AWS (federated)
aws sso login --profile income-adfs

# Configure kubectl for EKS
aws eks update-kubeconfig --region ap-southeast-1 --name <CLUSTER_NAME> --profile income-adfs
```

---

## Step 2: Build & Push Docker Image

```bash
# Set variables (update with IT team provided values)
export AWS_ACCOUNT_ID=<your-account-id>
export ECR_REPO=voice-quality-evaluator
export AWS_REGION=ap-southeast-1

# Login to ECR
aws ecr get-login-password --region $AWS_REGION --profile income-adfs | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
docker build -t $ECR_REPO .

# Tag image
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

---

## Step 3: Update Kubernetes Manifests

Edit `k8s/deployment.yaml`:
```yaml
image: <AWS_ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/voice-quality-evaluator:latest
serviceAccountName: <SERVICE_ACCOUNT_FROM_IT>
```

---

## Step 4: Deploy to EKS

```bash
# Set namespace
export NAMESPACE=voice-analysis

# Apply manifests
kubectl apply -f k8s/configmap.yaml -n $NAMESPACE
kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/service.yaml -n $NAMESPACE

# Check deployment status
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE
```

---

## Step 5: Access the Application

```bash
# Get LoadBalancer URL
kubectl get svc voice-quality-evaluator -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Open the URL in your browser to access the application.

---

## Troubleshooting

```bash
# Check pod logs
kubectl logs -f deployment/voice-quality-evaluator -n $NAMESPACE

# Describe pod for errors
kubectl describe pod -l app=voice-quality-evaluator -n $NAMESPACE

# Restart deployment
kubectl rollout restart deployment/voice-quality-evaluator -n $NAMESPACE
```

---

## File Structure

```
voice_analysis/
├── Dockerfile
├── .dockerignore
├── app.py
├── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── DEPLOYMENT.md (this file)
└── IT_INFRASTRUCTURE_REQUEST.md
```
