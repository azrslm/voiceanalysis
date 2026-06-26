"""
AWS Lambda: Transcription Agent (MP3/WAV in S3 -> parsed transcript payload)

This file is UI-free and can be deployed as a standalone Lambda handler.

Recommended input: S3 bucket+key for an audio file.
This handler will start an AWS Transcribe job and (optionally) wait for completion
to return the parsed transcript payload.

NOTE (AWS best practice):
- AWS Transcribe is asynchronous. For long calls, prefer an event-driven design:
  Start job -> EventBridge on completion -> parse -> store -> trigger evaluation.
This handler supports a `wait_for_completion` mode for simplicity / Step Functions.
"""

import base64
import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional

import boto3


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _parse_event_payload(event: Any) -> Dict[str, Any]:
    """Parse payload from API Gateway/Function URL (body) or direct invoke."""
    if not isinstance(event, dict):
        return {}

    # EventBridge: S3 Object Created events (detail.bucket.name + detail.object.key)
    # We normalize it into {bucket, key, ...} so the handler can be used directly as an EventBridge target.
    if event.get("source") == "aws.s3" and isinstance(event.get("detail"), dict):
        detail = event["detail"]
        bucket_name = None
        obj_key = None
        etag = None

        bucket_obj = detail.get("bucket")
        if isinstance(bucket_obj, dict):
            bucket_name = bucket_obj.get("name")
        elif isinstance(bucket_obj, str):
            bucket_name = bucket_obj

        object_obj = detail.get("object")
        if isinstance(object_obj, dict):
            obj_key = object_obj.get("key")
            etag = object_obj.get("etag") or object_obj.get("eTag")
        elif isinstance(object_obj, str):
            obj_key = object_obj

        if isinstance(obj_key, str):
            obj_key = urllib.parse.unquote_plus(obj_key)

        if bucket_name and obj_key:
            # Deterministic job name (prevents duplicate jobs on EventBridge retries).
            seed = f"{bucket_name}/{obj_key}|{etag or ''}"
            job_name = f"va_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"

            normalized: Dict[str, Any] = {"bucket": bucket_name, "key": obj_key, "job_name": job_name}
            # Best practice for EventBridge pipeline: don't wait inside Lambda.
            normalized["wait_for_completion"] = False
            return normalized

    # Generic EventBridge: pass through detail for custom events / other AWS sources.
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


def _infer_media_format_from_key(key: str) -> str:
    ext = os.path.splitext(str(key or ""))[1].lower().lstrip(".")
    if ext in {"wav", "mp3", "mp4", "flac", "ogg", "amr", "webm"}:
        return ext
    if ext == "m4a":
        return "mp4"
    return "wav"


def _http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _format_timestamp(seconds: float) -> str:
    minutes = int(float(seconds or 0.0) // 60)
    secs = int(float(seconds or 0.0) % 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_time_range(start: float, end: float) -> str:
    return f"{_format_timestamp(start)} → {_format_timestamp(end)}"


def _parse_transcribe_json(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse AWS Transcribe JSON into:
      - transcript_text: str (with speaker tags if available)
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

        # Infer which diarized speaker is CSO (best-effort)
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

    # Fallback: simple transcript
    transcript_text = ""
    try:
        transcript_text = str((results.get("transcripts") or [{}])[0].get("transcript") or "")
    except Exception:
        transcript_text = ""
    return {"use_channel_id": False, "transcript_text": transcript_text, "segments": [], "timing": None}


def handler(event: Any, context: Any) -> Dict[str, Any]:
    payload = _parse_event_payload(event)

    bucket = str(payload.get("bucket") or "").strip()
    key = str(payload.get("key") or "").lstrip("/").strip()
    if not bucket or not key:
        return _api_response(400, {"error": "Missing required fields: bucket, key"})

    job_name = str(payload.get("job_name") or "").strip() or f"va_{uuid.uuid4().hex}"
    language_code = str(payload.get("language_code") or os.getenv("DEFAULT_LANGUAGE_CODE") or "en-US").strip()
    use_channel_id = bool(payload.get("use_channel_id") or False)
    custom_vocabulary = str(payload.get("custom_vocabulary") or os.getenv("DEFAULT_CUSTOM_VOCABULARY") or "").strip() or None

    try:
        max_speakers = int(payload.get("max_speakers") or os.getenv("DEFAULT_MAX_SPEAKERS") or 2)
    except Exception:
        max_speakers = 2
    max_speakers = max(2, min(10, max_speakers))

    wait_for_completion = bool(payload.get("wait_for_completion") if "wait_for_completion" in payload else True)
    max_wait_seconds = int(payload.get("max_wait_seconds") or 840)  # 14 min default

    # Optional: store parsed transcript JSON to S3 (recommended for chaining)
    out_bucket = str(payload.get("output_bucket") or os.getenv("OUTPUT_BUCKET") or "").strip() or None
    out_prefix = str(payload.get("output_prefix") or os.getenv("OUTPUT_PREFIX") or "voice_analysis/").strip().strip("/") + "/"

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
    transcribe = boto3.client("transcribe", region_name=region)
    s3 = boto3.client("s3", region_name=region) if out_bucket else None

    media_format = str(payload.get("media_format") or "").strip().lower() or _infer_media_format_from_key(key)
    s3_uri = f"s3://{bucket}/{key}"

    settings: Dict[str, Any] = {}
    if use_channel_id:
        settings["ChannelIdentification"] = True
    else:
        settings["ShowSpeakerLabels"] = True
        settings["MaxSpeakerLabels"] = max_speakers
    if custom_vocabulary:
        settings["VocabularyName"] = custom_vocabulary

    logger.info(
        "TranscriptionAgent: start job=%s s3_uri=%s media_format=%s lang=%s channel_id=%s max_speakers=%s wait=%s",
        job_name,
        s3_uri,
        media_format,
        language_code,
        use_channel_id,
        max_speakers,
        wait_for_completion,
    )

    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=media_format,
            LanguageCode=language_code,
            Settings=settings,
        )
    except Exception as e:
        # EventBridge/S3 is at-least-once delivery. If we use a deterministic job_name,
        # retries may hit "job name already exists". Treat this as success (idempotent).
        err_code = None
        try:
            err_code = getattr(e, "response", {}).get("Error", {}).get("Code")
        except Exception:
            err_code = None

        is_conflict = (
            err_code == "ConflictException"
            or e.__class__.__name__ == "ConflictException"
            or "ConflictException" in str(e)
            or "already exists" in str(e).lower()
        )

        if is_conflict:
            try:
                existing = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                existing_status = existing["TranscriptionJob"]["TranscriptionJobStatus"]
                logger.info("TranscriptionAgent: job already exists job=%s status=%s", job_name, existing_status)
            except Exception:
                logger.info("TranscriptionAgent: job already exists job=%s", job_name)
        else:
            logger.exception("TranscriptionAgent: failed to start job: %s", e)
            return _api_response(500, {"error": "Failed to start transcription job", "detail": str(e)})

    if not wait_for_completion:
        # Persist a lightweight manifest for tracking (optional).
        if out_bucket and s3 is not None:
            try:
                manifest_key = f"{out_prefix}jobs/{job_name}.json"
                s3.put_object(
                    Bucket=out_bucket,
                    Key=manifest_key,
                    Body=json.dumps(
                        {
                            "job_name": job_name,
                            "status": "STARTED",
                            "bucket": bucket,
                            "key": key,
                            "s3_uri": s3_uri,
                            "language_code": language_code,
                            "media_format": media_format,
                            "use_channel_id": use_channel_id,
                            "max_speakers": max_speakers,
                            "started_at_epoch": int(time.time()),
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                    ContentType="application/json",
                )
            except Exception:
                logger.exception("TranscriptionAgent: failed to write job manifest to S3")

        return _api_response(
            200,
            {
                "job_name": job_name,
                "status": "STARTED",
                "bucket": bucket,
                "key": key,
            },
        )

    # Poll until completion or timeout
    start = time.time()
    poll_s = 2.0
    last_status = ""
    while True:
        elapsed = time.time() - start
        if elapsed > max_wait_seconds:
            return _api_response(
                202,
                {
                    "job_name": job_name,
                    "status": "IN_PROGRESS",
                    "message": f"Timed out waiting for completion after {max_wait_seconds}s. Re-invoke with job_name to fetch.",
                },
            )

        # Respect Lambda remaining time (leave buffer)
        try:
            if context is not None:
                remaining = int(getattr(context, "get_remaining_time_in_millis")() or 0)
                if remaining and remaining < 10_000:
                    return _api_response(
                        202,
                        {
                            "job_name": job_name,
                            "status": "IN_PROGRESS",
                            "message": "Lambda nearing timeout; re-invoke to fetch result.",
                        },
                    )
        except Exception:
            pass

        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if status != last_status:
            logger.info("TranscriptionAgent: job=%s status=%s", job_name, status)
            last_status = status

        if status == "COMPLETED":
            uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            raw = _http_get_json(uri)
            parsed = _parse_transcribe_json(raw)

            result = {
                "job_name": job_name,
                "status": "COMPLETED",
                "bucket": bucket,
                "key": key,
                "transcript_text": parsed.get("transcript_text", ""),
                "segments": parsed.get("segments", []),
                "timing": parsed.get("timing"),
                "use_channel_id": parsed.get("use_channel_id", False),
            }

            if out_bucket and s3 is not None:
                out_key = f"{out_prefix}transcripts/{job_name}.json"
                s3.put_object(
                    Bucket=out_bucket,
                    Key=out_key,
                    Body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
                    ContentType="application/json",
                )
                result["parsed_transcript_s3_bucket"] = out_bucket
                result["parsed_transcript_s3_key"] = out_key

            return _api_response(200, result)

        if status == "FAILED":
            reason = job["TranscriptionJob"].get("FailureReason", "Unknown")
            return _api_response(500, {"job_name": job_name, "status": "FAILED", "reason": reason})

        time.sleep(poll_s)
        poll_s = min(8.0, poll_s + 0.5)


# AWS Lambda expects `handler` as entrypoint (e.g., lambda_transcription_agent.handler)
lambda_handler = handler

