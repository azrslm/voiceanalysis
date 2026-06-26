"""
AWS Lambda: Evaluation Agent (transcript payload -> evaluation report)

This file is UI-free and can be deployed as a standalone Lambda handler.

Inputs supported:
  1) Direct transcript text:
      - transcript_text: "..."
      - transcript_id: optional
      - segments: optional (list) (for downstream evidence timeframe use)
      - timing: optional

  2) Parsed transcript JSON in S3 (recommended chaining from transcription agent):
      - parsed_transcript_s3_bucket
      - parsed_transcript_s3_key
      - transcript_id: optional

Outputs:
  - evaluation report JSON (report.to_dict())
  - optionally writes evaluation JSON to S3 if OUTPUT_BUCKET is set
"""

import base64
import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional

import boto3

from orchestrator import create_orchestrator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _parse_event_payload(event: Any) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    if "detail" in event and isinstance(event.get("detail"), dict):
        return event["detail"]
    if "body" in event:
        body = event.get("body")
        if event.get("isBase64Encoded") and isinstance(body, str):
            try:
                body = base64.b64decode(body).decode("utf-8")
            except Exception:
                body = ""
        if isinstance(body, dict):
            return body
        if isinstance(body, str) and body.strip():
            try:
                parsed = json.loads(body)
                return parsed if isinstance(parsed, dict) else {"items": parsed}
            except Exception:
                return {"raw_body": body}
    return event


def _api_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": int(status_code),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "OPTIONS,GET,POST",
        },
        "body": _json_dumps(body or {}),
    }


def _read_s3_json(s3, bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def _http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _format_timestamp(seconds: float) -> str:
    minutes = int(float(seconds or 0.0) // 60)
    secs = int(float(seconds or 0.0) % 60)
    return f"{minutes:02d}:{secs:02d}"


def _parse_transcribe_json(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse AWS Transcribe JSON into:
      - transcript_text: str
      - segments: List[{speaker, raw_speaker, start, end, text}]
      - timing: {start_time, end_time, duration} | None
      - use_channel_id: bool
    """
    results = response_json.get("results") or {}
    use_channel_id = bool("channel_labels" in results)

    # Channel identification
    if use_channel_id and "channel_labels" in results:
        channel_labels = results["channel_labels"]
        channels = channel_labels.get("channels", [])

        all_segments = []
        call_start = None
        call_end = None

        for channel in channels:
            channel_label = channel.get("channel_label", "")
            items = channel.get("items", [])

            if channel_label == "ch_0":
                speaker_role = "CSO"
            elif channel_label == "ch_1":
                speaker_role = "Customer"
            else:
                speaker_role = f"Speaker_{channel_label}"

            current_segment = []
            segment_start = None
            segment_end = None

            for item in items:
                if item.get("type") == "pronunciation":
                    if segment_start is None:
                        segment_start = float(item.get("start_time", 0))
                    segment_end = float(item.get("end_time", segment_start))
                    current_segment.append(item["alternatives"][0]["content"])
                elif item.get("type") == "punctuation":
                    current_segment.append(item["alternatives"][0]["content"])

                if current_segment and segment_end:
                    call_start = segment_start if call_start is None else min(call_start, segment_start)
                    call_end = segment_end if call_end is None else max(call_end, segment_end)

            if current_segment and segment_start is not None:
                text = " ".join(current_segment)
                text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
                all_segments.append(
                    {
                        "speaker": speaker_role,
                        "raw_speaker": channel_label,
                        "start": segment_start,
                        "end": segment_end or segment_start,
                        "text": text,
                    }
                )

        all_segments.sort(key=lambda x: x.get("start", 0.0))
        transcript_lines = []
        for seg in all_segments:
            ts = _format_timestamp(seg.get("start", 0.0))
            transcript_lines.append(f"{ts} **{seg.get('speaker','')}**: {seg.get('text','')}")

        timing = None
        if call_start is not None and call_end is not None:
            timing = {"start_time": call_start, "end_time": call_end, "duration": max(0.0, call_end - call_start)}

        return {
            "use_channel_id": True,
            "transcript_text": "\n\n".join(transcript_lines),
            "segments": all_segments,
            "timing": timing,
        }

    # Speaker diarization
    if "speaker_labels" in results:
        speaker_labels = results["speaker_labels"]
        items = results.get("items", [])

        segments = speaker_labels.get("segments", [])
        transcript_lines = []
        segments_info = []

        call_start = None
        call_end = None
        item_index = 0

        tmp_segments = []
        for seg in segments:
            raw_speaker = seg.get("speaker_label", "")
            start_time = float(seg.get("start_time", 0.0))
            end_time = float(seg.get("end_time", start_time))
            call_start = start_time if call_start is None else min(call_start, start_time)
            call_end = end_time if call_end is None else max(call_end, end_time)

            seg_text = []
            while item_index < len(items):
                item = items[item_index]
                if item.get("type") == "punctuation":
                    seg_text.append(item["alternatives"][0]["content"])
                    item_index += 1
                    continue
                if "start_time" in item:
                    item_start = float(item.get("start_time", 0.0))
                    if item_start < end_time:
                        seg_text.append(item["alternatives"][0]["content"])
                        item_index += 1
                    else:
                        break
                else:
                    item_index += 1

            text = " ".join(seg_text)
            text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            tmp_segments.append({"raw_speaker": raw_speaker, "start": start_time, "end": end_time, "text": text})

        # Infer CSO speaker (best-effort)
        def _infer_cso_raw_speaker(segs):
            if not segs:
                return "spk_0"
            lookahead = segs[:12]
            patterns = [
                "thank you for calling",
                "my name is",
                "this is",
                "how may i help",
                "how can i help",
                "you are speaking with",
                "you're speaking with",
                "hotline",
                "good morning",
                "good afternoon",
                "good evening",
                "for verification",
                "may i have",
                "could i have",
                "nric",
                "ic number",
            ]
            scores = {}
            for s in lookahead:
                raw = s.get("raw_speaker", "")
                t = (s.get("text", "") or "").lower()
                if not raw or not t:
                    continue
                score = scores.get(raw, 0)
                for p in patterns:
                    if p in t:
                        score += 2
                score += 1
                scores[raw] = score
            if scores:
                return max(scores.items(), key=lambda kv: kv[1])[0]
            return segs[0].get("raw_speaker") or "spk_0"

        cso_raw = _infer_cso_raw_speaker(tmp_segments)
        for seg in tmp_segments:
            raw = seg.get("raw_speaker", "")
            speaker_role = "CSO" if raw == cso_raw else "Customer"
            ts = _format_timestamp(seg.get("start", 0.0))
            transcript_lines.append(f"{ts} **{speaker_role}**: {seg.get('text','')}")
            segments_info.append(
                {
                    "speaker": speaker_role,
                    "raw_speaker": raw,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", seg.get("start", 0.0)),
                    "text": seg.get("text", ""),
                }
            )

        timing = None
        if call_start is not None and call_end is not None:
            timing = {"start_time": call_start, "end_time": call_end, "duration": max(0.0, call_end - call_start)}

        return {
            "use_channel_id": False,
            "transcript_text": "\n\n".join(transcript_lines),
            "segments": segments_info,
            "timing": timing,
        }

    transcript_text = ""
    try:
        transcript_text = str((results.get("transcripts") or [{}])[0].get("transcript") or "")
    except Exception:
        transcript_text = ""
    return {"use_channel_id": False, "transcript_text": transcript_text, "segments": [], "timing": None}


def handler(event: Any, context: Any) -> Dict[str, Any]:
    payload = _parse_event_payload(event)

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
    s3 = boto3.client("s3", region_name=region)
    transcribe = boto3.client("transcribe", region_name=region)

    # If invoked by EventBridge Transcribe completion, fetch + parse transcript automatically.
    evt_job = str(payload.get("TranscriptionJobName") or payload.get("transcriptionJobName") or "").strip()
    evt_status = str(payload.get("TranscriptionJobStatus") or payload.get("transcriptionJobStatus") or "").strip().upper()

    transcript_text = str(payload.get("transcript_text") or "").strip()
    transcript_id = str(payload.get("transcript_id") or "").strip() or None

    segments = payload.get("segments")
    timing = payload.get("timing")

    parsed_bucket = str(payload.get("parsed_transcript_s3_bucket") or "").strip()
    parsed_key = str(payload.get("parsed_transcript_s3_key") or "").strip()
    job_name = str(payload.get("job_name") or "").strip()

    # Optional persist targets
    out_bucket = str(payload.get("output_bucket") or os.getenv("OUTPUT_BUCKET") or "").strip() or None
    out_prefix = str(payload.get("output_prefix") or os.getenv("OUTPUT_PREFIX") or "voice_analysis/").strip().strip("/") + "/"

    if evt_job and (not transcript_text) and not (parsed_bucket and parsed_key):
        if evt_status and evt_status != "COMPLETED":
            return _api_response(200, {"ok": True, "job_name": evt_job, "status": evt_status})

        logger.info("EvaluationAgent: triggered by Transcribe completion job=%s", evt_job)
        job = transcribe.get_transcription_job(TranscriptionJobName=evt_job)
        transcript_uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        raw = _http_get_json(transcript_uri)
        parsed = _parse_transcribe_json(raw)
        transcript_text = str(parsed.get("transcript_text") or "").strip()
        segments = parsed.get("segments")
        timing = parsed.get("timing")
        job_name = evt_job
        transcript_id = transcript_id or job_name

        # Persist parsed transcript for traceability (optional)
        if out_bucket:
            t_key = f"{out_prefix}transcripts/{job_name}.json"
            s3.put_object(
                Bucket=out_bucket,
                Key=t_key,
                Body=json.dumps(
                    {
                        "job_name": job_name,
                        "use_channel_id": bool(parsed.get("use_channel_id")),
                        "transcript_text": transcript_text,
                        "segments": segments,
                        "timing": timing,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                ContentType="application/json",
            )
            parsed_bucket = out_bucket
            parsed_key = t_key

    if (not transcript_text) and parsed_bucket and parsed_key:
        parsed = _read_s3_json(s3, parsed_bucket, parsed_key)
        transcript_text = str(parsed.get("transcript_text") or "").strip()
        segments = segments if segments is not None else parsed.get("segments")
        timing = timing if timing is not None else parsed.get("timing")
        job_name = job_name or str(parsed.get("job_name") or "").strip()
        transcript_id = transcript_id or job_name or str(parsed.get("transcript_id") or "").strip() or None

    if not transcript_text:
        return _api_response(
            400,
            {"error": "Provide transcript_text OR parsed_transcript_s3_bucket + parsed_transcript_s3_key"},
        )

    transcript_id = transcript_id or job_name or "EVAL"
    run_parallel = bool(payload.get("parallel") if "parallel" in payload else True)

    bedrock_model_id = str(payload.get("bedrock_model_id") or os.getenv("BEDROCK_MODEL_ID") or "").strip() or None

    logger.info("EvaluationAgent: transcript_id=%s parallel=%s model_id=%s", transcript_id, run_parallel, bedrock_model_id or "")

    orch = create_orchestrator(profile_name=None, region_name=region, model_id=bedrock_model_id)
    report = orch.evaluate_transcript(transcript=transcript_text, transcript_id=transcript_id, parallel=run_parallel)
    report_dict = report.to_dict()

    # Optionally persist output to S3 (recommended for downstream)
    if out_bucket:
        out_key = f"{out_prefix}evaluations/{transcript_id}.json"
        logger.info("EvaluationAgent: writing evaluation to s3://%s/%s", out_bucket, out_key)
        s3.put_object(
            Bucket=out_bucket,
            Key=out_key,
            Body=json.dumps(
                {
                    **report_dict,
                    # Optional: pass-through transcript metadata (helps later joins)
                    "transcript_id": transcript_id,
                    "job_name": job_name,
                    "timing": timing,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            ContentType="application/json",
        )

        return _api_response(
            200,
            {
                "transcript_id": transcript_id,
                "evaluation_s3_bucket": out_bucket,
                "evaluation_s3_key": out_key,
                "parsed_transcript_s3_bucket": parsed_bucket or "",
                "parsed_transcript_s3_key": parsed_key or "",
                "overall_score_pct": report_dict.get("percentage_score"),
                "overall_rating": report_dict.get("overall_rating"),
            },
        )

    # Otherwise return full report in response
    # NOTE: For large payloads, prefer S3 output.
    return _api_response(
        200,
        {
            "transcript_id": transcript_id,
            "report": report_dict,
            "segments": segments,
            "timing": timing,
        },
    )


lambda_handler = handler

