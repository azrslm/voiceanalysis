## AWS Serverless Deployment Guide
### Voice Analysis Pipeline (Lambda + EventBridge) — No UI

This project’s **production** deployment is an event-driven backend pipeline:
- **AWS Transcribe** → creates transcripts asynchronously
- **AWS Lambda** → parses transcripts and runs **Bedrock** multi-agent evaluation
- **Amazon EventBridge** → orchestration via events
- **Amazon S3** → stores parsed transcripts, evaluations, and curated analytics datasets for **Tableau**

Streamlit/EKS artifacts exist in this repo as a **separate optional approach** (see “Approach B” at the end).

---

## Prerequisites (platform/DevOps)
- [ ] AWS accounts/environments defined (dev/uat/prod or dev+prod)
- [ ] Region confirmed (code defaults to `ap-southeast-1`)
- [ ] S3 buckets created:
  - [ ] Input audio bucket (call recordings)
  - [ ] Output/data-lake bucket (transcripts/evaluations/analytics)
- [ ] Bedrock model access approved (see `IT_INFRASTRUCTURE_REQUEST.md`)
- [ ] EventBridge bus/rules + Lambda execution roles provisioned (prefer IaC)

---

## Build Lambda deployment zip (local or CI)
PowerShell (from repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip

.\voiceanalys\scripts\build_lambda_zip.ps1
```

Output:
- `voiceanalys\dist\voice_analysis_lambda.zip`

---

## Deploy the Lambda functions (IaC preferred)

### Lambda A — Start Transcription (API-triggered)
- **Function name**: `voice-start-transcription`
- **Handler**: `lambda_handlers.start_transcription.handler`
- **Timeout**: ~30s
- **IAM**: `transcribe:StartTranscriptionJob`
- **Invocation**: API Gateway or Lambda Function URL (IAM auth recommended)
- **Optional (de-duplication)**:
  - Env var: `VOICE_ANALYSIS_JOBS_TABLE=<dynamodb-table>`
  - IAM: `dynamodb:UpdateItem` on that table

### Lambda B — Evaluate on Transcribe Complete (EventBridge-triggered)
- **Function name**: `voice-evaluate-transcript`
- **Handler**: `lambda_handlers.evaluate_transcript.handler`
- **Timeout**: up to 900s
- **Memory**: 1024MB+ (tune)
- **Env vars**:
  - `OUTPUT_BUCKET` (required)
  - `OUTPUT_PREFIX` (default `voice_analysis/`)
  - `EVENT_BUS_NAME` (default `default`)
  - Optional: `BEDROCK_MODEL_ID`, `INCLUDE_IMPROVEMENT_SUMMARY`
  - Optional roster: `AGENT_ROSTER_PATH` (recommended: S3 CSV)
- **Optional (de-duplication)**:
  - Env var: `VOICE_ANALYSIS_JOBS_TABLE=<dynamodb-table>`
  - IAM: `dynamodb:UpdateItem` on that table

---

## EventBridge wiring (required)

### Rule 1 — Transcribe completion → Evaluate Lambda
- **Event pattern**:
  - `source`: `aws.transcribe`
  - `detail-type`: `Transcribe Job State Change`
  - `detail.TranscriptionJobStatus`: `COMPLETED` (optionally include `FAILED`)

---

## Test plan (dev)
1. Upload a sample audio file to the input bucket.
2. Call the Start Transcription endpoint with:

```json
{
  "bucket": "YOUR_AUDIO_BUCKET",
  "key": "path/to/audio.wav",
  "language_code": "en-SG",
  "use_channel_id": false,
  "max_speakers": 2
}
```

3. Verify outputs in the output bucket:
- Parsed transcript JSON: `voice_analysis/transcripts/{job_name}.json`
- Evaluation JSON: `voice_analysis/evaluations/{transcript_id}.json`

4. If analytics publication is implemented, verify curated datasets:
- `voice_analysis/analytics/`

---

## References
- `AWS_INFRASTRUCTURE.md`: AWS resource design + Tableau data approach
- `SDP_VOICE_ANALYSIS_AWS.md`: end-to-end design + delivery plan
- `voiceanalys/README_LAMBDA.md`: handler wiring + env vars (implementation-level)

---

## Approach B (optional): Streamlit UI (local or EKS)
This repo also contains a Streamlit-based POC UI (`app.py`). Use this for demos/user testing only.

### Run locally
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

### Containerize and deploy to EKS (POC)
Artifacts:
- `Dockerfile` (Streamlit container)
- `k8s/` manifests (`deployment.yaml`, `service.yaml`, `configmap.yaml`)

High-level steps:
1. Build and push the container image to ECR
2. Update `k8s/deployment.yaml` image reference + IRSA service account
3. Apply Kubernetes manifests in the target namespace

For infra requirements, see `IT_INFRASTRUCTURE_REQUEST.md` → “Approach B — Streamlit UI on EKS”.
