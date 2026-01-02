# IT Team Infrastructure Request
## Voice Quality Evaluator - EKS Deployment

**Application:** AI-Powered Voice Quality Evaluator (Streamlit)  
**Purpose:** Deploy for user testing  
**Date:** 2026-01-02

---

## 1. EKS Cluster Access

| Requirement | Details |
|------------|---------|
| **EKS Cluster** | Access to an existing cluster OR provision a new one |
| **Namespace** | Dedicated namespace (e.g., `voice-analysis`) |
| **kubectl Access** | Kubeconfig file to connect to the cluster |

---

## 2. Container Registry (ECR)

| Requirement | Details |
|------------|---------|
| **ECR Repository** | Create repository: `voice-quality-evaluator` |
| **Push Access** | IAM permissions to push Docker images to ECR |

---

## 3. AWS IAM Permissions for Application

The application needs access to these AWS services. Please set up **IAM Roles for Service Accounts (IRSA)**:

| AWS Service | Permissions Required | Purpose |
|-------------|---------------------|---------|
| **S3** | `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, `s3:DeleteObject` | Audio file storage |
| **Transcribe** | `transcribe:StartTranscriptionJob`, `transcribe:GetTranscriptionJob` | Speech-to-text |
| **Bedrock** | `bedrock:InvokeModel` | AI evaluation (Claude 3.5 Sonnet) |

**S3 Bucket:** `genai-aws-poc` (path: `voice_analysis/voice_transcript/`)  
**Region:** `ap-southeast-1`  
**Bedrock Model:** `anthropic.claude-3-5-sonnet-20240620-v1:0`

---

## 4. Networking

| Requirement | Details |
|------------|---------|
| **LoadBalancer** | Application Load Balancer (ALB) OR Classic LoadBalancer |
| **Ingress Controller** | AWS Load Balancer Controller (if using Ingress) |
| **Security Group** | Allow inbound on port 80/443 from user network |
| **DNS (Optional)** | Subdomain like `voice-eval.internal.company.com` |

---

## 5. Resource Allocation

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 500m | 1 core |
| **Memory** | 1Gi | 2Gi |
| **Replicas** | 1 | 2 |
| **Storage** | None (stateless) | - |

---

## 6. What I Will Provide

Once IT prepares the above, I will provide:
- ✅ Dockerfile (containerization)
- ✅ Kubernetes manifests (deployment.yaml, service.yaml)
- ✅ Application source code
- ✅ Deployment instructions

---

## 7. Information Needed from IT

Please provide back:

1. **ECR Repository URI**: `xxxxxxxxxxxx.dkr.ecr.ap-southeast-1.amazonaws.com/voice-quality-evaluator`
2. **EKS Cluster Name**: `_________________`
3. **Kubernetes Namespace**: `_________________`
4. **IRSA Service Account Name**: `_________________`
5. **Kubeconfig or AWS SSO details** for kubectl access
6. **LoadBalancer type preference**: ALB / NLB / ClusterIP

---

## Contact
For questions about this request, contact: [Your Name/Email]
