## AWS Infrastructure — Voice Analysis (Serverless, No UI)

### Scope (what we are building)
- **In scope**: Event-driven AWS pipeline to (1) transcribe call audio via **AWS Transcribe**, (2) parse/standardize the transcript, (3) evaluate call quality via **AWS Bedrock** multi-agent evaluation, (4) publish outputs for **Tableau** consumption.
- **Out of scope (for this document)**: Streamlit / web UI, EKS-hosted UI services (documented separately as an optional approach).

This design aligns with the existing Lambda handlers in `voiceanalys/lambda_handlers/`:
- `start_transcription.handler`
- `evaluate_transcript.handler`

### High-level architecture (recommended)
```
Caller audio files
   │
   │ (upload / landing)
   ▼
S3 (Input Audio Bucket)
   │
   │ 1) API call (Function URL / API Gateway)
   ▼
Lambda: voice-start-transcription
   │
   │ StartTranscriptionJob
   ▼
AWS Transcribe (async)
   │
   │ EventBridge: Transcribe Job State Change (COMPLETED)
   ▼
Lambda: voice-evaluate-transcript
   │
   │ - checks Transcribe completion + fetches transcript JSON
   │ - parses transcript + segments (checkpoint written to S3)
   │ - invokes Bedrock (multi-agent evaluation)
   │ - writes evaluation JSON to S3
   │ - emits EventBridge custom event: voice.evaluation.ready
   ▼
S3 (Output / Data Lake Bucket)
   │
   │ (analytics shaping)
   ├── Option A: Glue Crawler + Athena over curated Parquet/CSV
   └── Option B: Load to Redshift/Aurora for Tableau
   ▼
Tableau (dashboards + scheduled refresh)
```

### AWS services and required resources
#### Core compute & orchestration
- **AWS Lambda**
  - `voice-start-transcription` (sync API)
  - `voice-evaluate-transcript` (event-driven; parses + evaluates; long-running)
  - Optional: `voice-publish-analytics` (transform evaluation JSON → tabular datasets for Tableau)
- **Amazon EventBridge**
  - Event bus: `default` or a dedicated bus (recommended: `voice-analysis-bus`)
  - Rules:
    - Transcribe completion → `voice-evaluate-transcript`
    - (Optional) `voice.evaluation.ready` → `voice-publish-analytics`
- **AWS Transcribe**
  - Used in async mode; completion events drive downstream processing.

#### Storage & analytics
- **Amazon S3**
  - **Input bucket** (landing audio): `s3://<audio-input-bucket>/<prefix>/...`
  - **Output bucket** (parsed transcripts + evaluations + analytics datasets):
    - Parsed transcripts: `voice_analysis/transcripts/{job_name}.json`
    - Evaluations: `voice_analysis/evaluations/{transcript_id}.json`
    - Curated analytics (recommended): `voice_analysis/analytics/` (Parquet/CSV)
- **AWS Glue + Athena** (recommended for serverless Tableau data source)
  - Glue Data Catalog databases/tables for curated datasets
  - Glue crawlers or explicit table DDL for stable schema
  - Athena workgroup for Tableau queries
- **Alternative**: **Amazon Redshift** (for performance / governed BI)
  - Use COPY from S3 curated data or direct ingestion from a transform job.

#### Security, identity, and secrets
- **IAM**
  - Separate execution role per Lambda function (least privilege)
  - Resource policies for S3 buckets to allow Transcribe read access (and optionally Transcribe write)
- **KMS**
  - SSE-KMS for S3 buckets (recommended)
  - KMS key policies allowing Lambda roles and required service principals
- **Secrets Manager / SSM Parameter Store**
  - Store configuration values that should not be plain env vars (if any non-public config is required)
  - Note: Bedrock + Transcribe use IAM auth; no API keys required.

#### Observability and operations
- **CloudWatch Logs** for all Lambda functions
- **CloudWatch Alarms** (recommended)
  - Lambda errors/throttles
  - DLQ depth (if configured)
  - Transcribe failures (via EventBridge metrics / custom metrics)
- **DLQ / retry strategy** (recommended)
  - Lambda async destinations or SQS DLQs for `voice-evaluate-transcript` (and any analytics publisher)

### Lambda configuration (from code)
#### Function: `voice-start-transcription`
- **Handler**: `lambda_handlers.start_transcription.handler`
- **Timeout**: ~30s
- **Memory**: 256–512MB
- **IAM**: `transcribe:StartTranscriptionJob`
- **Optional idempotency**:
  - Env var: `VOICE_ANALYSIS_JOBS_TABLE` (DynamoDB table name)
  - If set, the handler uses DynamoDB to prevent duplicate Transcribe job starts.
- **Notes**:
  - Input payload expects `{bucket, key, ...}`.
  - The input S3 bucket must allow Transcribe service principal to read audio objects.

#### Function: `voice-evaluate-transcript`
- **Handler**: `lambda_handlers.evaluate_transcript.handler`
- **Timeout**: up to 900s (15 min) due to multiple Bedrock calls
- **Memory**: 1024MB+ (tune)
- **Env vars**:
  - `OUTPUT_BUCKET` (**required**)
  - `OUTPUT_PREFIX` (default `voice_analysis/`)
  - `EVENT_BUS_NAME` (default `default`)
  - `EMIT_EVALUATION_READY_EVENT` (default `true`)
  - `BEDROCK_MODEL_ID` (optional; defaults in code to Claude Sonnet)
  - `INCLUDE_IMPROVEMENT_SUMMARY` (optional; default false)
  - Optional cost controls:
    - `OPENING_VERIFICATION_MAX_SECONDS`
    - `EARLY_PHASE_BEDROCK_MODEL_ID`, `OPENING_GREETING_BEDROCK_MODEL_ID`, `VERIFICATION_BEDROCK_MODEL_ID`
  - Optional roster enrichment:
    - `AGENT_ROSTER_PATH` (recommended: `s3://.../agent_roster.csv`)
    - `AGENT_ROSTER_SHEET` (Excel only; avoid for Lambda unless dependencies are included)
- **Optional idempotency**:
  - Env var: `VOICE_ANALYSIS_JOBS_TABLE` (DynamoDB table name)
  - If set, the handler claims an evaluation lock to avoid duplicate Bedrock calls on retries.
- **IAM**:
  - `transcribe:GetTranscriptionJob` (when triggered by Transcribe completion events)
  - `s3:GetObject` for parsed transcript JSON
  - `s3:PutObject` for parsed transcript checkpoint (`voice_analysis/transcripts/{job_name}.json`)
  - `s3:PutObject` for evaluation output
  - `events:PutEvents` (if emitting evaluation-ready event)
  - `bedrock:InvokeModel` (bedrock-runtime)

### DynamoDB job registry (recommended for de-duplication)
EventBridge and most schedulers are **at-least-once** (duplicate deliveries can happen). To ensure you do not:
- start the same Transcribe job multiple times, and/or
- pay for duplicate Bedrock evaluations,
enable the optional DynamoDB “job registry”.

- **Env var**: `VOICE_ANALYSIS_JOBS_TABLE=<table-name>`
- **Table**: DynamoDB table with **partition key**:
  - `job_name` (String)
- **Required permissions** (on that table):
  - For start Lambda: `dynamodb:UpdateItem`
  - For evaluate Lambda: `dynamodb:UpdateItem`

The pipeline stores/updates metadata such as:
- `status` (e.g., `TRANSCRIBE_REQUESTED`, `TRANSCRIBE_STARTED`, `TRANSCRIBE_COMPLETED`, `EVALUATED`)
- `evaluation_status` (`EVALUATING`, `EVALUATED`, `FAILED_EVAL`)
- `parsed_transcript_s3_key`, `evaluation_s3_key`
- timestamps + last error

### Data outputs (what Tableau will consume)
The Lambda writes **nested JSON** evaluation outputs. Tableau typically needs **tabular** datasets.

Recommended approach:
- Create curated datasets (CSV/Parquet) in S3, then query with Athena (or load into Redshift).

Minimum curated tables:
- **`call_evaluations`** (1 row per call)
  - `transcript_id`, `evaluation_timestamp`, `detected_agent_name`, `detected_agent_canonical_name`, `detected_agent_match_score`
  - `percentage_score`, `overall_rating`
  - Optional: `raw_percentage_score`, `weight_sum`, etc.
- **`criteria_evaluations`** (many rows per call; 1 per criterion)
  - `transcript_id`, `agent_category`, `criteria_id`, `criteria_name`
  - `rating`, `score`, `max_score`, `weight`
  - `evidence` (string), `reasoning` (string)

### Environments and naming
Recommended separation:
- **AWS accounts**: `dev` / `uat` / `prod` (or at least dev + prod)
- **Resource naming**: prefix by environment, e.g. `va-dev-voice-evaluate-transcript`
- **Tagging**: `app=voice-analysis`, `env=dev|prod`, `owner=<team>`, `cost_center=<...>`

### Deployment and IaC expectations
DevOps should provision all infrastructure with IaC (Terraform/CloudFormation/CDK) including:
- Buckets (SSE-KMS, versioning, lifecycle, access logs)
- Lambda functions (deploy zip artifact), env vars, reserved concurrency
- EventBridge bus/rules, permissions
- IAM roles/policies
- Glue/Athena (if chosen) and curated dataset locations

See `voiceanalys/README_LAMBDA.md` for manual-console wiring reference; production should be IaC-driven.

