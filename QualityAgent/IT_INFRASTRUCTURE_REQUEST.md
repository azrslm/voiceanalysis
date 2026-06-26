## IT / Platform Infrastructure Request
### Voice Analysis — Two Supported Approaches

This project can be deployed in **two ways**:
- **Approach A (recommended / production)**: **AWS-ready serverless** pipeline (Lambda + EventBridge + S3 + Transcribe + Bedrock) producing Tableau datasets.
- **Approach B (optional / POC)**: **Streamlit UI** container deployed on EKS (uses the same AWS services under the hood).

**Application**: Voice Analysis (Transcribe → parse → Bedrock evaluation → Tableau datasets)  
**Purpose**: Provide automated call QA scoring + BI-ready outputs  
**Date**: 2026-01-30

---

## 1. AWS account + region

| Requirement | Details |
|------------|---------|
| **AWS Account(s)** | dev / uat / prod (or dev + prod minimum) |
| **Region** | `ap-southeast-1` (matches current defaults in code) |
| **Tagging standard** | app/env/owner/cost-center tags applied consistently |

---

## 2. S3 buckets (input + outputs + analytics)

| Requirement | Details |
|------------|---------|
| **Input audio bucket** | `s3://<AUDIO_INPUT_BUCKET>/<prefix>/...` (landing call recordings) |
| **Output/data-lake bucket** | `s3://<VOICE_OUTPUT_BUCKET>/voice_analysis/` |
| **Required prefixes** | `voice_analysis/transcripts/`, `voice_analysis/evaluations/`, `voice_analysis/analytics/` |
| **Encryption** | SSE-KMS recommended for all buckets |
| **Lifecycle/retention** | Define retention for audio, transcripts, evaluations, and curated analytics |

---

## 3. Lambda functions + IAM permissions (least privilege)

### Approach A — AWS-ready serverless (recommended)

We will deploy **two Lambda functions** (plus optional analytics transform). Please provision execution roles with minimum permissions:

| Function | Handler | Permissions (minimum) | Notes |
|---|---|---|---|
| **voice-start-transcription** | `lambda_handlers.start_transcription.handler` | `transcribe:StartTranscriptionJob` | Starts Transcribe job for an existing S3 object |
| **voice-evaluate-transcript** | `lambda_handlers.evaluate_transcript.handler` | `transcribe:GetTranscriptionJob`, `s3:GetObject`, `s3:PutObject`, `events:PutEvents`, `bedrock:InvokeModel` | Triggered on Transcribe completion; parses transcript + evaluates; emits `voice.evaluation.ready` |
| **(Optional) voice-publish-analytics** | TBD | `s3:GetObject`, `s3:PutObject` (+ Glue/Athena/Redshift perms if used) | Flattens evaluation JSON to curated datasets for Tableau |

**Bedrock model (default in code)**: `anthropic.claude-3-5-sonnet-20240620-v1:0`  
**Important**: Lambda should use IAM role creds (do not rely on `AWS_PROFILE`).

#### Optional: DynamoDB job registry (de-duplication)
To prevent duplicate Transcribe starts and duplicate Bedrock evaluations (EventBridge is at-least-once), optionally provision:

- **DynamoDB table**: `<VOICE_ANALYSIS_JOBS_TABLE>`
  - **Partition key**: `job_name` (String)
- **Env var on both Lambdas**: `VOICE_ANALYSIS_JOBS_TABLE=<table-name>`
- **IAM**: `dynamodb:UpdateItem` on that table

---

### Approach B — Streamlit UI on EKS (optional / POC)
If the Streamlit UI is required (e.g., for user testing), provision the following **in addition**:

| Requirement | Details |
|---|---|
| **EKS cluster / namespace** | Namespace e.g., `voice-analysis` |
| **ECR repository** | Repo for Streamlit container image |
| **IRSA service account** | Pod role with S3/Transcribe/Bedrock permissions |
| **Ingress / LoadBalancer** | Internal access on 80/443 |
| **ConfigMap / Secrets** | Env vars for bucket names, region, model id |

---

## 4. EventBridge wiring

| Requirement | Details |
|------------|---------|
| **Event bus** | Use `default` or provision dedicated bus (recommended: `voice-analysis-bus`) |
| **Rule: Transcribe completion** | `aws.transcribe` → `voice-evaluate-transcript` |
| **(Optional) rule: evaluation ready** | `voice.evaluation.ready` → analytics transform |

---

## 5. Networking (only if required by policy)

This solution can run Lambdas **without VPC attachment** (simplest).
If VPC attachment is required:
- Provide required **VPC endpoints** (at minimum S3; and Bedrock Runtime if restricted)
- Ensure Lambdas can access the Transcribe transcript artifact (current code downloads via `TranscriptFileUri` HTTPS)

---

## 6. Analytics for Tableau (data source provisioning)

Choose one:
- **Option A (serverless)**: Glue Data Catalog + Athena tables over curated S3 datasets
- **Option B (warehouse)**: Redshift tables populated from curated S3 datasets

Please confirm which option to provision for Tableau connectivity.

---

## 7. What the application team will provide

- ✅ Lambda deployment zip artifact (built from `voiceanalys/scripts/build_lambda_zip.ps1`)
- ✅ Handler names + environment variable list
- ✅ Event schemas and S3 output schemas
- ✅ Runbooks and test payloads

---

## 8. Information needed from IT / Platform / DevOps

1. **AWS account IDs** + environment mapping (dev/uat/prod)
2. **Approved AWS region(s)**
3. **S3 bucket names** for input audio and output/data-lake (and KMS key IDs if applicable)
4. **Event bus name** (`default` vs dedicated)
5. **Bedrock model approval** (model ID and any org-specific guardrails)
6. **Tableau integration option** (Athena vs Redshift) + connectivity requirements
7. **Preferred API exposure** for starting transcription (API Gateway vs Function URL) + auth requirement

---

## Contact
For questions about this request, contact: [Your Name/Email]
