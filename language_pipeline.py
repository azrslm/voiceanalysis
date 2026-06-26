"""
Language detection metadata helpers and Bedrock translation for evaluation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Subset of AWS Transcribe LanguageCode values (ap-southeast-1 / contact-center use).
# Note: en-SG is not a valid Transcribe code; use en-IN or en-US for Singapore English.
DEFAULT_TRANSCRIBE_LANGUAGE_OPTIONS: List[str] = [
    "en-US",
    "en-GB",
    "en-AU",
    "en-IN",
    "zh-CN",
    "zh-TW",
    "zh-HK",
    "ms-MY",
    "ta-IN",
    "hi-IN",
    "ja-JP",
    "ko-KR",
    "th-TH",
    "vi-VN",
    "id-ID",
    "fr-FR",
    "de-DE",
    "es-US",
    "pt-BR",
    "ar-SA",
]

# Full allow-list from AWS Transcribe StartTranscriptionJob validation.
AWS_VALID_TRANSCRIBE_LANGUAGE_CODES = frozenset(
    {
        "en-IE", "ar-AE", "pa-IN", "be-BY", "te-IN", "my-MM", "zh-HK", "zh-TW", "en-US",
        "uk-UA", "sw-KE", "gu-IN", "ta-IN", "en-AB", "ug-CN", "su-ID", "bn-IN", "hy-AM",
        "km-KH", "en-IN", "sl-SI", "ab-GE", "zh-CN", "es-MX", "ar-SA", "eu-ES", "en-ZA",
        "gd-GB", "cy-WL", "uz-UZ", "tl-PH", "so-SO", "sk-SK", "rw-RW", "ro-RO", "pl-PL",
        "no-NO", "mt-MT", "mr-IN", "mn-MN", "mk-MK", "lv-LV", "lt-LT", "is-IS", "hu-HU",
        "hr-HR", "ha-NG", "fi-FI", "et-ET", "bg-BG", "az-AZ", "th-TH", "tr-TR", "ru-RU",
        "pt-PT", "nl-NL", "it-IT", "id-ID", "ht-HT", "fr-FR", "es-ES", "de-DE", "sw-RW",
        "sw-TZ", "sr-RS", "ps-AF", "or-IN", "kn-IN", "ga-IE", "af-ZA", "wo-SN", "tt-RU",
        "sw-BI", "en-NZ", "ko-KR", "am-ET", "el-GR", "ba-RU", "hi-IN", "de-CH", "vi-VN",
        "cy-GB", "ml-IN", "ms-MY", "he-IL", "cs-CZ", "ka-GE", "si-LK", "gl-ES", "lg-IN",
        "kab-DZ", "fa-AF", "da-DK", "ne-NP", "en-AU", "zu-ZA", "mhr-RU", "ast-ES", "pt-BR",
        "en-WL", "sq-AL", "sw-UG", "ky-KG", "ckb-IQ", "bs-BA", "fa-IR", "kk-KZ", "ckb-IR",
        "sv-SE", "jv-ID", "ja-JP", "mi-NZ", "ca-ES", "es-US", "et-EE", "fr-CA", "en-GB",
    }
)

ENGLISH_LANGUAGE_PREFIX = "en"


def is_english_language_code(code: str | None) -> bool:
    return bool(code and str(code).lower().startswith(ENGLISH_LANGUAGE_PREFIX))


def filter_valid_language_options(options: List[str]) -> List[str]:
    valid: List[str] = []
    invalid: List[str] = []
    for code in options:
        if code in AWS_VALID_TRANSCRIBE_LANGUAGE_CODES:
            if code not in valid:
                valid.append(code)
        else:
            invalid.append(code)
    if invalid:
        logger.warning("Removed unsupported AWS Transcribe language codes: %s", invalid)
    return valid


def normalize_language_options(options: Optional[List[str]]) -> List[str]:
    cleaned = [str(code).strip() for code in (options or []) if str(code).strip()]
    source = cleaned or list(DEFAULT_TRANSCRIBE_LANGUAGE_OPTIONS)
    valid = filter_valid_language_options(source)
    if not valid:
        valid = filter_valid_language_options(list(DEFAULT_TRANSCRIBE_LANGUAGE_OPTIONS))
    return valid


def build_transcription_language_params(
    *,
    language_code: str = "en-US",
    auto_detect_language: bool = True,
    identify_multiple_languages: bool = False,
    language_options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build kwargs for AWS Transcribe start_transcription_job language settings.
    """
    options = normalize_language_options(language_options)

    # AWS automatic language identification requires at least two candidate languages.
    if len(options) < 2:
        options = normalize_language_options(None)

    if identify_multiple_languages:
        return {
            "IdentifyMultipleLanguages": True,
            "LanguageOptions": options,
        }

    if auto_detect_language:
        return {
            "IdentifyLanguage": True,
            "LanguageOptions": options,
        }

    fixed = language_code or "en-US"
    if fixed not in AWS_VALID_TRANSCRIBE_LANGUAGE_CODES:
        logger.warning("Unsupported fixed language %s; falling back to en-US", fixed)
        fixed = "en-US"
    return {"LanguageCode": fixed}


def extract_language_metadata(
    transcription_job: Dict[str, Any],
    transcript_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract detected language information from Transcribe job + transcript JSON."""
    job = transcription_job.get("TranscriptionJob", transcription_job) or {}
    detected: List[str] = []

    primary = str(job.get("LanguageCode") or "").strip()
    if primary:
        detected.append(primary)

    for code in job.get("LanguageCodes") or []:
        code_str = str(code).strip()
        if code_str:
            detected.append(code_str)

    for item in job.get("LanguageIdentification") or []:
        code_str = str(item.get("LanguageCode") or "").strip()
        if code_str:
            detected.append(code_str)

    if transcript_json:
        results = transcript_json.get("results") or {}
        result_lang = str(results.get("language_code") or "").strip()
        if result_lang:
            detected.append(result_lang)

        for item in results.get("language_identification") or []:
            code_str = str(item.get("language_code") or "").strip()
            if code_str:
                detected.append(code_str)

    unique = list(dict.fromkeys(detected))
    return {
        "detected_languages": unique,
        "primary_language": unique[0] if unique else (primary or None),
        "language_identification": job.get("LanguageIdentification") or [],
    }


@dataclass
class PreparedTranscript:
    original_transcript: str
    evaluation_transcript: str
    original_segments: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_segments: List[Dict[str, Any]] = field(default_factory=list)
    detected_languages: List[str] = field(default_factory=list)
    was_translated: bool = False
    translation_notes: str = ""


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def translate_transcript_to_english(
    transcript_text: str,
    *,
    config: Any,
    detected_languages: Optional[List[str]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> PreparedTranscript:
    """
    Translate transcript text to English using Bedrock while preserving structure.
    """
    original = (transcript_text or "").strip()
    langs = list(detected_languages or [])
    segs = list(segments or [])

    if not original:
        return PreparedTranscript(
            original_transcript="",
            evaluation_transcript="",
            original_segments=segs,
            evaluation_segments=[],
            detected_languages=langs,
            was_translated=False,
        )

    if langs and all(is_english_language_code(code) for code in langs):
        return PreparedTranscript(
            original_transcript=original,
            evaluation_transcript=original,
            original_segments=segs,
            evaluation_segments=[dict(seg) for seg in segs],
            detected_languages=langs,
            was_translated=False,
            translation_notes="Transcript already in English.",
        )

    try:
        from agents import BedrockConfig, create_bedrock_llm
    except ImportError:
        from evaluation.agents import BedrockConfig, create_bedrock_llm

    if not isinstance(config, BedrockConfig):
        config = BedrockConfig(
            profile_name=getattr(config, "profile_name", None),
            region_name=getattr(config, "region_name", None),
            model_id=getattr(config, "model_id", None),
            aws_access_key_id=getattr(config, "aws_access_key_id", None),
            aws_secret_access_key=getattr(config, "aws_secret_access_key", None),
        )

    llm = create_bedrock_llm(config)
    lang_hint = ", ".join(langs) if langs else "unknown / mixed"

    prompt = f"""You translate contact-center call transcripts for quality evaluation.

Detected language(s): {lang_hint}

Rules:
1. Translate the transcript to natural English.
2. Preserve timestamps, speaker labels (CSO, Customer, Agent, Speaker_*), and line order.
3. Do not summarize, omit, or add commentary.
4. If the transcript is already English, return it unchanged.
5. Return ONLY the translated transcript text.

Transcript:
{original}
"""

    try:
        response = llm.invoke(prompt)
        translated = _strip_code_fences(getattr(response, "content", str(response)))
    except Exception as exc:
        logger.error("Bedrock translation failed: %s", exc)
        translated = original

    was_translated = translated.strip() != original.strip()
    evaluation_segments = _translate_segments(segs, original, translated, was_translated)

    return PreparedTranscript(
        original_transcript=original,
        evaluation_transcript=translated,
        original_segments=segs,
        evaluation_segments=evaluation_segments,
        detected_languages=langs,
        was_translated=was_translated,
        translation_notes="Translated to English via Bedrock." if was_translated else "No translation required.",
    )


def _translate_segments(
    segments: List[Dict[str, Any]],
    original_transcript: str,
    translated_transcript: str,
    was_translated: bool,
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    if not was_translated:
        return [dict(seg) for seg in segments]

    # Best-effort line alignment when segment count matches translated lines.
    translated_lines = [
        line.strip()
        for line in translated_transcript.splitlines()
        if line.strip()
    ]
    if len(translated_lines) != len(segments):
        return [dict(seg) for seg in segments]

    out: List[Dict[str, Any]] = []
    for seg, line in zip(segments, translated_lines):
        updated = dict(seg)
        updated["text"] = _strip_speaker_prefix(line)
        updated["original_text"] = seg.get("text", "")
        out.append(updated)
    return out


def _strip_speaker_prefix(line: str) -> str:
    # "00:12 **CSO**: hello" -> "hello"
    return re.sub(r"^\d{2}:\d{2}\s+\*\*[^*]+\*\*:\s*", "", line).strip() or line.strip()


def prepare_transcript_for_evaluation(
    transcript_text: str,
    *,
    config: Any,
    detected_languages: Optional[List[str]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    translate_to_english: bool = True,
) -> PreparedTranscript:
    if not translate_to_english:
        original = (transcript_text or "").strip()
        segs = list(segments or [])
        return PreparedTranscript(
            original_transcript=original,
            evaluation_transcript=original,
            original_segments=segs,
            evaluation_segments=[dict(seg) for seg in segs],
            detected_languages=list(detected_languages or []),
            was_translated=False,
            translation_notes="Translation disabled.",
        )

    return translate_transcript_to_english(
        transcript_text,
        config=config,
        detected_languages=detected_languages,
        segments=segments,
    )
