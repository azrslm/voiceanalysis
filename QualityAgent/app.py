

import streamlit as st
import copy
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import subprocess
import shutil
import boto3
import time
import uuid
import requests
import os
import base64
import logging
import json
import math
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from aws_utils import create_aws_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


DEFAULT_AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
DEFAULT_AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
DEFAULT_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# AWS Pricing (ap-southeast-1 region)
TRANSCRIBE_PRICE_PER_MINUTE = 0.024  # USD per minute
BEDROCK_INPUT_PRICE_PER_1K = 0.003   # USD per 1K input tokens (Claude 3.5 Sonnet)
BEDROCK_OUTPUT_PRICE_PER_1K = 0.015  # USD per 1K output tokens (Claude 3.5 Sonnet)

# Where to store generated Excel reports in S3 (prefix/"folder")
DEFAULT_EXCEL_REPORT_S3_PREFIX = os.getenv("EXCEL_REPORT_S3_PREFIX", "voice_analysis/excel report/").rstrip("/") + "/"

# Model pricing for cost comparison (USD per 1K tokens).
# Note: Prices can vary by provider/region/account and may change over time.
MODEL_PRICING_USD_PER_1K = {
    # AWS Bedrock (Anthropic) - example rates used elsewhere in this app
    "Claude 3.5 Sonnet": {"input": 0.003, "output": 0.015},
    "Claude 3 Sonnet": {"input": 0.003, "output": 0.015},
    # OpenAI (converted from per-1M token list prices to per-1K)
    "GPT-4.1": {"input": 0.002, "output": 0.008},
    "GPT-5.1": {"input": 0.00125, "output": 0.01},
}

BIGTAPP_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "bigtapp_logo.png"


def _bigtapp_logo_data_uri() -> str:
    encoded = base64.b64encode(BIGTAPP_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


st.set_page_config(
    page_title="BigTapp | Voice Quality Evaluator",
    page_icon="🎙️",
    layout="wide"
)


st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* BigTapp Brand Colors */
    :root {
        --bigtapp-blue: #1D4ED8;
        --bigtapp-blue-light: #2563EB;
        --bigtapp-navy: #0B2E69;
        --bigtapp-navy-light: #15438F;
        --bigtapp-white: #FFFFFF;
        --bigtapp-ice: #F5F9FF;
        --bigtapp-grey: #EEF3FB;
        --bigtapp-border: #D7E3F5;
        --bigtapp-text: #102A43;
        --bigtapp-text-muted: #486581;
    
    }
    
    /* ===== MAIN APP BACKGROUND ===== */
    .stApp {
        background-color: var(--bigtapp-ice) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] {
        background-color: var(--bigtapp-ice) !important;
    }
    
    .stApp [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* ===== SIDEBAR - BIGTAPP BLUE ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2E69 0%, #15438F 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding-top: 1.5rem;
    }
    
    /* Sidebar Text - Light on dark background */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"]:hover {
        background-color: rgba(255,255,255,0.12) !important;
    }
    
    /* Sidebar Inputs - readable dark text on white fields */
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] input[type="password"],
    [data-testid="stSidebar"] .stTextInput input {
        background-color: #FFFFFF !important;
        color: #102A43 !important;
        border: 1px solid #D7E3F5 !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] input:focus {
        border-color: var(--bigtapp-blue) !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Sidebar Slider - Blue Track */
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: var(--bigtapp-blue) !important;
    }
    
    /* Sidebar Checkbox */
    [data-testid="stSidebar"] .stCheckbox > label > div:first-child {
        background-color: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    

    /* Sidebar expander labels and content */
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary {
        background-color: transparent !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: rgba(255,255,255,0.05) !important;
        border-top: 1px solid rgba(255,255,255,0.12) !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label {
        color: #EAF2FF !important;
    }

    [data-testid="stSidebar"] input::placeholder {
        color: #486581 !important;
    }

    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }
    
    /* ===== HEADER BANNER ===== */
    .bigtapp-header {
        background: linear-gradient(135deg, var(--bigtapp-navy) 0%, #15438F 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(11, 46, 105, 0.2);
    }
    
    .bigtapp-header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .bigtapp-logo {
        display: flex;
        align-items: center;
    }

    .bigtapp-logo img {
        height: 44px;
        width: auto;
        display: block;
        border-radius: 10px;
        background: #FFFFFF;
        padding: 6px 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }

    .bigtapp-sidebar-logo {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 1rem;
    }

    .bigtapp-sidebar-logo img {
        height: 52px;
        width: auto;
        display: inline-block;
        border-radius: 10px;
        background: #FFFFFF;
        padding: 8px 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    
    .bigtapp-header-divider {
        width: 1px;
        height: 32px;
        background: rgba(255,255,255,0.2);
        margin: 0 0.5rem;
    }
    
    .bigtapp-header h1 {
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    
    .bigtapp-header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .bigtapp-badge {
        background: var(--bigtapp-blue);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ===== CONTENT CARDS ===== */
    .content-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid var(--bigtapp-border);
        margin-bottom: 1.5rem;
    }
    
    .content-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--bigtapp-grey);
    }
    
    .content-card-header h3 {
        color: var(--bigtapp-navy);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .content-card-header .icon {
        font-size: 1.3rem;
    }
    
    /* ===== METRIC BOXES ===== */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid var(--bigtapp-border);
        text-align: center;
        transition: all 0.2s ease;
        border-top: 3px solid var(--bigtapp-blue);
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.1);
    }
    
    .metric-label {
        color: var(--bigtapp-text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: var(--bigtapp-navy);
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    .metric-value.score {
        color: var(--bigtapp-blue);
        font-size: 2rem;
    }
    
    /* ===== REPORT CARDS ===== */
    .report-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid var(--bigtapp-border);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .report-card:hover {
        border-color: var(--bigtapp-blue);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
    }
    
    /* ===== EVIDENCE BOX ===== */
    .evidence-box {
        background: linear-gradient(135deg, #F5F9FF, #EAF2FF);
        border-left: 3px solid var(--bigtapp-blue);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.75rem;
        font-style: italic;
        color: var(--bigtapp-navy);
    }
    
    /* ===== BADGES ===== */
    .rating-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .badge-Expected { background: #E8F5E9; color: #2E7D32; }
    .badge-Desirable { background: #C8E6C9; color: #1B5E20; }
    .badge-Basic { background: #EAF2FF; color: #1D4ED8; }
    .badge-Undesirable { background: #FFEBEE; color: #C62828; }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--bigtapp-navy);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ===== TRANSCRIPT BOX ===== */
    .transcript-box {
        background: var(--bigtapp-grey);
        border-radius: 10px;
        padding: 1.25rem;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid var(--bigtapp-border);
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.85rem;
    }
    
    .speaker-label {
        font-weight: 700;
        color: var(--bigtapp-blue);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }
    
    /* ===== INPUT FIELDS (Main Area) ===== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: white !important;
        border: 1px solid var(--bigtapp-border) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--bigtapp-blue) !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--bigtapp-grey);
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        color: var(--bigtapp-text-muted);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
        color: white !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1D4ED8, #2563EB) !important;
        border-radius: 4px;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed var(--bigtapp-border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--bigtapp-blue) !important;
        background: #F5F9FF !important;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }
    
    /* ===== HIDE MAIN HEADER (we use custom) ===== */
    .main-header {
        display: none;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bigtapp-grey);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #9DB4DA;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--bigtapp-blue);
    }
</style>
""", unsafe_allow_html=True)

# Custom Header with BigTapp branding
_bigtapp_logo_uri = _bigtapp_logo_data_uri()
st.markdown(
    f"""
<div class="bigtapp-header">
    <div class="bigtapp-header-left">
        <div class="bigtapp-logo">
            <img src="{_bigtapp_logo_uri}" alt="BigTapp logo" />
        </div>
    </div>
    <div class="bigtapp-header-right">
        <h1>Voice Quality Evaluator</h1>
        <div class="bigtapp-badge">AI Powered</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# BigTapp Logo in Sidebar (top left corner above AWS Configuration)
st.sidebar.markdown(
    f"""
<div class="bigtapp-sidebar-logo">
    <img src="{_bigtapp_logo_uri}" alt="BigTapp logo" />
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("🔐 AWS Configuration")

with st.sidebar.expander("AWS Settings", expanded=True):
    aws_access_key_id = st.text_input(
        "AWS Access Key ID",
        key="aws_access_key_id",
        value=DEFAULT_AWS_ACCESS_KEY_ID,
    ).strip()
    aws_secret_access_key = st.text_input(
        "AWS Secret Access Key",
        key="aws_secret_access_key",
        value=DEFAULT_AWS_SECRET_ACCESS_KEY,
        type="password",
    ).strip()
    aws_region = st.text_input(
        "AWS Region",
        value=DEFAULT_AWS_REGION,
        key="aws_region"
    )
    s3_bucket_name = st.text_input(
        "S3 Bucket Name",
        key="s3_bucket",
        value=DEFAULT_S3_BUCKET
    )

with st.sidebar.expander("👤 Agent Roster (Skillset)", expanded=False):
    roster_default = os.getenv("AGENT_ROSTER_PATH", "").strip()
    roster_path = st.text_input(
        "Roster file path (CSV/XLSX)",
        key="agent_roster_path",
        value=roster_default,
        help="Local path to roster CSV/XLSX (e.g., C:/.../CC Staff Skillset_Updated 15 Jan 2026.xlsx).",
    ).strip()

    roster_sheet_default = os.getenv("AGENT_ROSTER_SHEET", "").strip()
    roster_sheet = st.text_input(
        "Roster sheet (optional)",
        key="agent_roster_sheet",
        value=roster_sheet_default,
        help="Excel sheet name or 0-based index. Leave blank to use the first sheet.",
    ).strip()

    if roster_path:
        # Make available to orchestrator/roster loader (no S3 required)
        os.environ["AGENT_ROSTER_PATH"] = roster_path
        if roster_path.split("\\")[-1].startswith("~$") or roster_path.split("/")[-1].startswith("~$"):
            st.warning("That file looks like an Excel temp/lock file (~$...). Please select the actual roster .xlsx.")
    else:
        # Do not override if user clears the input
        os.environ.pop("AGENT_ROSTER_PATH", None)

    if roster_sheet:
        os.environ["AGENT_ROSTER_SHEET"] = roster_sheet
    else:
        os.environ.pop("AGENT_ROSTER_SHEET", None)

with st.sidebar.expander("Transcription Settings", expanded=False):
    max_speakers = st.number_input("Max Speakers (for Diarization)", min_value=2, max_value=10, value=2)
    
    # Channel Identification - best for stereo audio with agent/customer on separate channels
    use_channel_id = st.checkbox(
        "Use Channel Identification", 
        value=False,
        help="Enable if audio is STEREO with agent on one channel and customer on another. This gives 100% accurate speaker identification."
    )
    
    auto_detect_language = st.checkbox(
        "Auto-detect language",
        value=True,
        help="Let AWS Transcribe detect the spoken language automatically.",
    )
    identify_multiple_languages = st.checkbox(
        "Detect multiple languages in one call",
        value=False,
        help="Optional. Enable only for calls that switch languages mid-call. Uses AWS multi-language mode.",
    )
    language_code = st.selectbox(
        "Fixed language (when auto-detect is off)",
        options=["en-US", "en-GB", "en-AU", "en-IN", "zh-CN", "ms-MY", "ta-IN", "hi-IN"],
        index=0,
        disabled=auto_detect_language or identify_multiple_languages,
        help="Used only when auto-detect and multi-language detection are both disabled.",
    )
    
    # Custom vocabulary (optional)
    custom_vocabulary = st.text_input(
        "Custom Vocabulary Name (optional)",
        value="",
        help="Enter the name of a pre-created AWS Transcribe custom vocabulary"
    )
    
    # Show speaker labels info
    if use_channel_id:
        st.info("📢 **Channel ID mode**\n\n- **ch_0 (Left)**: CSO / Agent\n- **ch_1 (Right)**: Customer / PH")
    else:
        st.info(f"📢 **Speaker diarization mode**\n\nUp to **{max_speakers}** speakers will be detected.")


st.sidebar.markdown("---")
st.sidebar.header("🎚️ Audio Settings")
prop_decrease = st.sidebar.slider("Noise Reduction Strength", 0.0, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("📊 Evaluation Settings")
run_parallel = st.sidebar.checkbox("Run Agents in Parallel", value=True)
translate_to_english = st.sidebar.checkbox(
    "Translate transcript to English before evaluation",
    value=True,
    help="Uses Bedrock to translate non-English or mixed-language transcripts before quality scoring.",
)


def upload_to_s3(file_obj, bucket, object_name, s3_client):
    """Upload file to S3"""
    logger.info(f"Starting S3 upload: Bucket={bucket}, Key={object_name}")
    try:
        s3_client.upload_fileobj(file_obj, bucket, object_name)
        logger.info("S3 upload successful")
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        st.error(f"Error uploading to S3: {e}")
        return False


def _evaluate_transcript_with_translation(
    *,
    transcript_text,
    segments,
    filename,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
    run_parallel,
    translate_to_english,
    detected_languages=None,
):
    """Translate (if needed) then run multi-agent quality evaluation."""
    from orchestrator import create_orchestrator
    from agents import BedrockConfig
    from language_pipeline import prepare_transcript_for_evaluation

    config = BedrockConfig(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
        model_id=DEFAULT_BEDROCK_MODEL_ID,
    )

    prepared = prepare_transcript_for_evaluation(
        transcript_text,
        config=config,
        detected_languages=detected_languages or [],
        segments=segments,
        translate_to_english=translate_to_english,
    )

    orchestrator = create_orchestrator(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
        model_id=DEFAULT_BEDROCK_MODEL_ID,
    )

    report = orchestrator.evaluate_transcript(
        transcript=prepared.evaluation_transcript,
        transcript_id=filename,
        parallel=run_parallel,
    )

    if report:
        report.original_transcript = prepared.original_transcript
        report.evaluation_transcript = prepared.evaluation_transcript
        report.detected_languages = prepared.detected_languages
        report.was_translated = prepared.was_translated
        report.translation_notes = prepared.translation_notes

    return report, prepared, orchestrator



def list_s3_audio_files(bucket, prefix, s3_client):
    """
    List audio files from S3 bucket with given prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: Folder path prefix (e.g., 'voice_analysis/voice_transcript/')
        s3_client: boto3 S3 client
    
    Returns:
        List of dicts with file info: {key, size, last_modified, filename}
    """
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith(audio_extensions):
                    files.append({
                        'key': key,
                        'filename': key.split('/')[-1],
                        'size': obj['Size'],
                        'size_mb': round(obj['Size'] / (1024 * 1024), 2),
                        'last_modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        logger.info(f"Found {len(files)} audio files in s3://{bucket}/{prefix}")
        return files
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        return []


def download_s3_file(bucket, key, s3_client):
    """
    Download file from S3 to memory buffer.
    
    Args:
        bucket: S3 bucket name
        key: Full S3 object key
        s3_client: boto3 S3 client
    
    Returns:
        BytesIO buffer with file contents, or None on error
    """
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        logger.info(f"Downloaded s3://{bucket}/{key}")
        return buffer
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return None


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def _ffmpeg_convert_to_pcm16_wav(
    audio_bytes: bytes,
    *,
    input_suffix: str = ".wav",
    mono: bool = True,
    target_sr: int | None = None,
) -> bytes:
    """
    Convert audio bytes to WAV PCM_16 using ffmpeg.
    Requires ffmpeg on PATH (Docker image already installs it).
    """
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on PATH")

    ffmpeg = shutil.which("ffmpeg")
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix or ".wav") as tmp_in:
            tmp_in.write(audio_bytes)
            in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            out_path = tmp_out.name

        cmd = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            in_path,
        ]
        if mono:
            cmd += ["-ac", "1"]
        if target_sr:
            cmd += ["-ar", str(int(target_sr))]
        cmd += ["-c:a", "pcm_s16le", out_path]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg conversion failed: {stderr or 'unknown error'}")

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass


def _infer_transcribe_media_format_from_key(key: str) -> str:
    """
    Infer AWS Transcribe MediaFormat from an S3 key/filename.
    AWS expects values like: wav, mp3, mp4, flac, ogg, amr, webm (availability can vary).
    """
    ext = os.path.splitext(str(key or ""))[1].lower().lstrip(".")
    if ext in {"wav", "mp3", "mp4", "flac", "ogg", "amr", "webm"}:
        return ext
    # Common alias: m4a is an MP4 container
    if ext == "m4a":
        return "mp4"
    # Default to wav for safety
    return "wav"


def _load_audio_from_bytes(audio_bytes: bytes, *, mono: bool = True, suffix: str = ".wav"):
    """
    Robust audio loader for BytesIO/UploadedFile content.
    Tries in-memory load first; falls back to a temp file so librosa/audioread backends can handle more WAV variants.
    """
    if not audio_bytes:
        raise ValueError("Empty audio bytes")

    # First attempt: in-memory (fast)
    try:
        return librosa.load(io.BytesIO(audio_bytes), sr=None, mono=mono)
    except Exception:
        # Try ffmpeg conversion if available (handles malformed WAV headers / uncommon codecs)
        if _ffmpeg_available():
            converted = _ffmpeg_convert_to_pcm16_wav(
                audio_bytes,
                input_suffix=suffix or ".wav",
                mono=mono,
                target_sr=None,
            )
            return librosa.load(io.BytesIO(converted), sr=None, mono=mono)

    # Fallback: temp file path (enables audioread/ffmpeg backends when available)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        return librosa.load(tmp_path, sr=None, mono=mono)
    except Exception as e:
        # If we got here, decoding failed and we likely lack a backend (NoBackendError) or the WAV is truly invalid.
        if "NoBackendError" in type(e).__name__ and not _ffmpeg_available():
            raise RuntimeError(
                "Audio decoding failed because no backend is available. "
                "Install ffmpeg (recommended) or run the app via the provided Docker image which includes ffmpeg."
            ) from e
        raise
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def process_s3_audio_file(
    bucket, 
    file_key, 
    s3_client, 
    transcribe_client,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
    max_speakers=2,
    use_channel_id=False,
    language_code='en-US',
    auto_detect_language=True,
    identify_multiple_languages=False,
    custom_vocabulary=None,
    prop_decrease=0.75,
    run_parallel=True,
    translate_to_english=True,
):
    """
    Process a single S3 audio file through the full pipeline.
    
    Returns:
        dict with results: {filename, transcript, evaluation_report, error}
    """
    filename = file_key.split('/')[-1]
    result = {
        'filename': filename,
        'key': file_key,
        'transcript': None,
        'evaluation_report': None,
        'score': None,
        'rating': None,
        'error': None,
        'cost_estimation': None,  # Cost estimation for this file
        'ai_suggestion_for_improvement': None,  # Separate AI agent output (coaching summary)
        # Additional data for evidence voice clips and metrics
        'audio_data': None,
        'sr': None,
        'segments': None,
        'timing': None,
        'hold_analysis': None,
        'hold_summary': None,
        'cso_wpm': None,
        'audio_duration_seconds': None
    }
    
    try:
        # Download audio file
        audio_buffer = download_s3_file(bucket, file_key, s3_client)
        if audio_buffer is None:
            result['error'] = "Failed to download file"
            return result

        # Policy:
        # - If the ORIGINAL file is WAV: send directly to AWS Transcribe (skip noise reduction entirely)
        # - Only attempt noise reduction when the ORIGINAL file is MP3
        # - If MP3 decode/processing fails: fall back to direct AWS Transcribe on the original S3 object
        original_media_format = _infer_transcribe_media_format_from_key(file_key)

        transcribe_s3_uri = None
        media_format = original_media_format
        temp_key = None
        reduced_noise_y = None
        sr = None
        audio_duration_seconds = None

        if original_media_format == "wav":
            # Direct transcription from original WAV object
            transcribe_s3_uri = f"s3://{bucket}/{file_key}"
            media_format = "wav"
        elif original_media_format == "mp3":
            # MP3:
            # - In diarization mode: attempt decode + noise reduce + upload clean PCM WAV (Transcribe-friendly)
            # - In channel-id mode: DO NOT downmix to mono for transcription; transcribe the original MP3 to preserve stereo channels
            if use_channel_id:
                # Preserve stereo channels for Channel Identification
                transcribe_s3_uri = f"s3://{bucket}/{file_key}"
                media_format = "mp3"

                # Still try local decode + noise reduction (mono) for UI snippets/hold analysis where possible
                try:
                    audio_buffer.seek(0)
                    y, sr = _load_audio_from_bytes(
                        audio_buffer.getvalue(),
                        mono=True,
                        suffix=os.path.splitext(filename)[1].lower() or ".mp3",
                    )
                    reduced_noise_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=False)
                    audio_duration_seconds = len(reduced_noise_y) / sr if sr and sr > 0 else None
                except Exception as decode_err:
                    logger.warning(f"{filename}: MP3 local decode/analysis failed (will still transcribe original): {decode_err}")
            else:
                try:
                    audio_buffer.seek(0)
                    y, sr = _load_audio_from_bytes(
                        audio_buffer.getvalue(),
                        mono=True,
                        suffix=os.path.splitext(filename)[1].lower() or ".mp3",
                    )

                    reduced_noise_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=False)

                    processed_buffer = io.BytesIO()
                    sf.write(processed_buffer, reduced_noise_y, sr, format='WAV', subtype='PCM_16')
                    processed_buffer.seek(0)

                    temp_key = f"temp_processed/{uuid.uuid4()}.wav"
                    if not upload_to_s3(processed_buffer, bucket, temp_key, s3_client):
                        raise RuntimeError("Failed to upload processed audio for transcription")

                    transcribe_s3_uri = f"s3://{bucket}/{temp_key}"
                    media_format = "wav"
                    audio_duration_seconds = len(reduced_noise_y) / sr if sr and sr > 0 else None
                except Exception as decode_err:
                    logger.warning(
                        f"{filename}: MP3 decode/processing failed; falling back to direct AWS Transcribe on original S3 object. "
                        f"Reason: {type(decode_err).__name__}: {decode_err}"
                    )
                    transcribe_s3_uri = f"s3://{bucket}/{file_key}"
                    media_format = original_media_format
        else:
            # Other formats: direct transcription
            transcribe_s3_uri = f"s3://{bucket}/{file_key}"
            media_format = original_media_format

        # Transcribe
        job_name = f"batch_transcription_{uuid.uuid4()}"
        transcript_result = transcribe_audio(
            transcribe_s3_uri,
            job_name, 
            transcribe_client, 
            max_speakers=max_speakers,
            use_channel_id=use_channel_id,
            language_code=language_code,
            auto_detect_language=auto_detect_language,
            identify_multiple_languages=identify_multiple_languages,
            custom_vocabulary=custom_vocabulary,
            media_format=media_format,
        )
        
        # Cleanup temp file
        if temp_key:
            try:
                s3_client.delete_object(Bucket=bucket, Key=temp_key)
            except Exception:
                pass
        
        if transcript_result is None:
            result['error'] = "Transcription failed"
            return result
        
        transcript_text, segments, timing, language_info = transcript_result
        result['transcript'] = transcript_text
        result['detected_languages'] = (language_info or {}).get('detected_languages', [])
        result['language_info'] = language_info or {}
        result['segments'] = segments
        result['timing'] = timing
        
        # Store audio data for voice clip playback (only if we successfully decoded/processed locally)
        if reduced_noise_y is not None and sr is not None:
            result['audio_data'] = reduced_noise_y
            result['sr'] = sr

        # Calculate audio duration: prefer waveform-derived duration, else use transcript timing duration
        if audio_duration_seconds is None:
            try:
                audio_duration_seconds = float((timing or {}).get("duration", 0.0)) if timing else 0.0
            except Exception:
                audio_duration_seconds = 0.0
        result['audio_duration_seconds'] = audio_duration_seconds
        
        # Run hold detection (hold music vs dead silence)
        # Need to temporarily set audio_duration_seconds for detect_hold_segments
        old_duration = st.session_state.get('audio_duration_seconds')
        st.session_state['audio_duration_seconds'] = audio_duration_seconds
        
        if reduced_noise_y is not None and sr is not None:
            hold_analysis = detect_hold_segments(
                segments or [],
                reduced_noise_y,
                sr,
                min_gap=10.0
            )
            result['hold_analysis'] = hold_analysis
            result['hold_summary'] = summarize_hold_segments(hold_analysis)
        else:
            result['hold_analysis'] = []
            result['hold_summary'] = {"hold_music": 0.0, "dead_silence": 0.0}
        
        # Restore original value
        if old_duration is not None:
            st.session_state['audio_duration_seconds'] = old_duration
        
        # Calculate words per minute
        result['cso_wpm'] = compute_words_per_minute(segments or [], timing)
        
        # Translate (if needed) and run quality evaluation
        report, prepared, orchestrator = _evaluate_transcript_with_translation(
            transcript_text=transcript_text,
            segments=segments,
            filename=filename,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            run_parallel=run_parallel,
            translate_to_english=translate_to_english,
            detected_languages=(language_info or {}).get("detected_languages", []),
        )
        result['original_transcript'] = prepared.original_transcript
        result['evaluation_transcript'] = prepared.evaluation_transcript
        result['was_translated'] = prepared.was_translated
        result['translation_notes'] = prepared.translation_notes
        
        if report:
            result['evaluation_report'] = report
            result['score'] = round(report.percentage_score, 1)
            result['rating'] = report.overall_rating

            # Generate "AI Suggestion for Improvement" using a separate AI agent (no hard-coded mapping)
            try:
                from agents import ImprovementSummaryAgent

                def _normalize_text(t: str) -> str:
                    if not t:
                        return ""
                    t = t.lower()
                    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t).split())

                def _find_timeframe_for_evidence(ev_text: str):
                    if not ev_text or not segments:
                        return None
                    ev_norm = _normalize_text(ev_text)
                    if not ev_norm:
                        return None
                    for seg in segments or []:
                        seg_text = seg.get("text", "")
                        seg_norm = _normalize_text(seg_text)
                        if ev_norm and seg_norm and ev_norm in seg_norm:
                            try:
                                start = float(seg.get("start", 0.0))
                                end = float(seg.get("end", start))
                                return format_time_range(start, end)
                            except Exception:
                                return None
                    return None

                # Reuse the same Bedrock LLM instance created for evaluation agents
                llm = None
                try:
                    agents_map = getattr(orchestrator, "agents", None) or {}
                    # In rare cases (e.g., orchestrator reused / partial init), agents_map can be empty here.
                    # Try to (re)initialize to make sure agents exist.
                    if not agents_map and hasattr(orchestrator, "initialize_agents"):
                        try:
                            orchestrator.initialize_agents()
                            agents_map = getattr(orchestrator, "agents", None) or {}
                        except Exception:
                            agents_map = getattr(orchestrator, "agents", None) or {}

                    # Prefer a "full-context" agent's LLM so optional cheap early-phase models
                    # (opening/verification) don't degrade the coaching summary quality.
                    any_agent = (
                        agents_map.get("soft_skills")
                        or agents_map.get("enquiry_resolution")
                        or agents_map.get("wrap_up")
                        or next(iter(agents_map.values()))
                    )
                    llm = getattr(any_agent, "llm", None)
                except Exception:
                    llm = None

                # Fallback: create a fresh Bedrock LLM for the summary agent so we never skip
                if llm is None:
                    try:
                        from agents import create_bedrock_llm, BedrockConfig

                        cfg = BedrockConfig(
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            region_name=aws_region,
                            model_id=DEFAULT_BEDROCK_MODEL_ID,
                        )
                        llm = create_bedrock_llm(cfg)
                        logger.info(f"{filename}: Summary agent using fresh Bedrock LLM instance (fallback).")
                    except Exception as e:
                        logger.warning(f"{filename}: Summary agent could not create Bedrock LLM fallback: {e}")
                        llm = None

                if llm is not None:
                    logger.info(f"{filename}: Generating AI Suggestion for Improvement (summary agent)...")
                    summary_agent = ImprovementSummaryAgent(llm)

                    evaluations = []
                    issues = []
                    for agent_result in (getattr(report, "agent_results", {}) or {}).values():
                        for ev in getattr(agent_result, "evaluations", []) or []:
                            try:
                                score = float(getattr(ev, "score", 0.0))
                                max_score = float(getattr(ev, "max_score", 0.0))
                                weight = float(getattr(ev, "weight", 0.0))
                            except Exception:
                                score, max_score, weight = 0.0, 0.0, 0.0

                            rating = getattr(ev, "rating", "") or ""
                            gap = max(0.0, max_score - score)
                            is_issue = rating in ("Undesirable", "Basic") or gap > 1e-6

                            sev = 0 if rating == "Undesirable" else (1 if rating == "Basic" else 2)
                            ev_texts = list(getattr(ev, "evidence", None) or [])
                            ev_one = (ev_texts[0] if ev_texts else "") or ""
                            tf = _find_timeframe_for_evidence(ev_one) if ev_one else None
                            evidence_obj = {"quote": ev_one, "timeframe": tf} if ev_one else None

                            item = {
                                "category": getattr(agent_result, "agent_name", "") or "",
                                "criterion": getattr(ev, "criteria_name", "") or "",
                                "rating": rating,
                                "score": score,
                                "max_score": max_score,
                                "weight": weight,
                                "evidence": [evidence_obj] if evidence_obj else [],
                                "ai_finding": (getattr(ev, "reasoning", "") or "").strip(),
                            }
                            evaluations.append(item)
                            if is_issue:
                                issues.append({**item, "_sort": (sev, -gap, -weight)})

                    issues.sort(key=lambda x: x.get("_sort", (9, 0, 0)))
                    top_issues = []
                    for item in issues[:10]:
                        item.pop("_sort", None)
                        top_issues.append(item)

                    findings_payload = {
                        "transcript_id": getattr(report, "transcript_id", "") or filename,
                        "overall_score_pct": float(getattr(report, "percentage_score", 0.0) or 0.0),
                        "overall_rating": getattr(report, "overall_rating", "") or "",
                        # Full evaluation outputs (AI findings + evidence) for summarization
                        "evaluations": evaluations,
                        # Optional focus list for biggest gaps (kept small)
                        "issues": top_issues,
                    }

                    logger.info(
                        f"{filename}: Summary agent input prepared "
                        f"(evaluations={len(evaluations)}, issues={len(top_issues)})."
                    )
                    ai_summary = summary_agent.generate(findings_payload)
                    if ai_summary:
                        result["ai_suggestion_for_improvement"] = ai_summary
                        logger.info(f"{filename}: Summary agent completed (chars={len(ai_summary)}).")
                    else:
                        logger.warning(f"{filename}: Summary agent returned empty output.")
                else:
                    logger.warning(f"{filename}: Summary agent skipped (could not access Bedrock LLM instance).")
            except Exception as e:
                logger.error(f"Error generating AI Suggestion for Improvement for {filename}: {e}")

            # Upload per-file Excel report to S3 (voice_analysis/excel report/)
            try:
                from orchestrator import DetailedReportGenerator

                excel_bytes = DetailedReportGenerator.generate_excel_report(
                    report,
                    segments=segments or [],
                ).getvalue()
                ts = _timestamp_for_filename(getattr(report, "evaluation_timestamp", ""))
                base_name = _safe_s3_key_part(os.path.splitext(filename)[0])
                object_key = f"{DEFAULT_EXCEL_REPORT_S3_PREFIX}evaluation_report_{base_name}_{ts}.xlsx"
                uploaded = _maybe_upload_excel_bytes_to_s3(
                    excel_bytes=excel_bytes,
                    bucket=bucket,
                    object_key=object_key,
                    s3_client=s3_client,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_region=aws_region,
                )
                if uploaded:
                    result["excel_report_s3_key"] = object_key
            except Exception as e:
                logger.error(f"Error generating/uploading per-file Excel report for {filename}: {e}")
        
        # Calculate cost estimation
        if audio_duration_seconds and transcript_text:
            result['cost_estimation'] = calculate_total_processing_cost(
                audio_duration_seconds,
                transcript_text
            )
        
        return result
        
    except Exception as e:
        # Full traceback in logs; many audio decoding errors have empty str(e)
        logger.exception(f"Error processing {filename}")
        msg = str(e).strip()
        result['error'] = f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}"
        return result


def generate_batch_results_excel(results):
    """
    Generate Excel report from batch processing results.
    
    Args:
        results: List of result dicts from process_s3_audio_file
    
    Returns:
        BytesIO buffer with Excel file
    """
    import pandas as pd
    
    # Summary sheet data
    summary_rows = []
    for r in results:
        cost_data = r.get('cost_estimation', {}) or {}
        token_info = cost_data.get('bedrock', {}) if cost_data else {}
        est_total_tokens = token_info.get('total_tokens', 'N/A') if token_info else 'N/A'

        improvement_text = r.get("ai_suggestion_for_improvement") or ""
        if not improvement_text:
            improvement_text = "AI suggestion unavailable (summary agent not run or failed)."

        report = r.get("evaluation_report")
        detected_name = getattr(report, "detected_agent_name", "") if report else ""
        matched_name = getattr(report, "detected_agent_canonical_name", "") if report else ""
        skills = getattr(report, "detected_agent_skills", []) if report else []
        match_score = getattr(report, "detected_agent_match_score", 0.0) if report else 0.0
        skills_str = ", ".join([str(s) for s in (skills or []) if str(s).strip()])

        summary_rows.append({
            'File Name': r['filename'],
            'S3 Key': r['key'],
            'Detected Agent Name': detected_name or "",
            'Matched Agent Name': matched_name or "",
            'Agent Skills (LI/GI/HI)': skills_str,
            'Name Match Score': round(float(match_score or 0.0), 3) if report else "",
            'Call Classification': getattr(report, "call_classification", "") if report else "",
            'Call Subject': getattr(report, "call_subject", "") if report else "",
            'Customer Issues': " | ".join(getattr(report, "call_issues", []) or []) if report else "",
            'Score (%)': r['score'] if r['score'] else 'N/A',
            'Rating': r['rating'] if r['rating'] else 'N/A',
            'Status': 'Success' if r['evaluation_report'] else f"Error: {r['error']}",
            # Cost columns removed from Excel summary (UI still shows cost)
            'Token Size (Est.)': est_total_tokens,
            'AI Suggestion for Improvement': improvement_text,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Detailed results per file
    detailed_rows = []
    for r in results:
        if r['evaluation_report']:
            report = r['evaluation_report']
            segments = r.get('segments') or []
            report_weight_sum = float(getattr(report, "weight_sum", 0.0) or 0.0)
            detected_name = getattr(report, "detected_agent_name", "") or ""
            matched_name = getattr(report, "detected_agent_canonical_name", "") or ""
            skills = getattr(report, "detected_agent_skills", []) or []
            skills_str = ", ".join([str(s) for s in skills if str(s).strip()])
            match_score = float(getattr(report, "detected_agent_match_score", 0.0) or 0.0)

            def _short_criteria_id(cid: str) -> str:
                return cid.split(".", 1)[1] if isinstance(cid, str) and "." in cid else (cid or "")

            def _normalize_text(t: str) -> str:
                if not t:
                    return ""
                t = t.lower()
                # keep alnum + spaces only
                return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t).split()

            def _find_timeframe_for_evidence(ev_text: str):
                if not ev_text or not segments:
                    return None
                ev_tokens = _normalize_text(ev_text)
                if not ev_tokens:
                    return None
                ev_norm = " ".join(ev_tokens)
                for seg in segments:
                    seg_text = seg.get("text", "")
                    seg_norm = " ".join(_normalize_text(seg_text))
                    if ev_norm and seg_norm and ev_norm in seg_norm:
                        try:
                            start = float(seg.get("start", 0.0))
                            end = float(seg.get("end", start))
                            return format_time_range(start, end)
                        except Exception:
                            return None
                return None

            for agent_name, agent_result in report.agent_results.items():
                for eval in agent_result.evaluations:
                    evidence_str = " | ".join(eval.evidence) if getattr(eval, "evidence", None) else ""
                    timeframes = []
                    for ev_txt in (getattr(eval, "evidence", None) or []):
                        tf = _find_timeframe_for_evidence(ev_txt)
                        timeframes.append(tf or "")
                    evidence_timeframe = " | ".join([t for t in timeframes if t]) if any(timeframes) else ""

                    # Make score math transparent in the Excel export:
                    # - Normalized Score: score / max_score
                    # - Effective Weight (%): weight normalized to sum to 100 across included criteria
                    # - Contribution (%): Effective Weight (%) * Normalized Score
                    try:
                        _w = float(getattr(eval, "weight", 0.0) or 0.0)
                        _s = float(getattr(eval, "score", 0.0) or 0.0)
                        _m = float(getattr(eval, "max_score", 0.0) or 0.0)
                    except Exception:
                        _w, _s, _m = 0.0, 0.0, 0.0

                    normalized_score = None
                    effective_weight_pct = None
                    contribution_pct = None
                    if _w > 0.0 and _m > 0.0 and report_weight_sum > 0.0:
                        normalized_score = _s / _m
                        effective_weight_pct = (_w / report_weight_sum) * 100.0
                        contribution_pct = effective_weight_pct * normalized_score

                    detailed_rows.append({
                        'File Name': r['filename'],
                        'Detected Agent Name': detected_name,
                        'Matched Agent Name': matched_name,
                        'Agent Skills (LI/GI/HI)': skills_str,
                        'Name Match Score': round(match_score, 3),
                        'Agent Category': agent_result.agent_name,
                        'Criteria ID': _short_criteria_id(eval.criteria_id),
                        'Criteria Name': eval.criteria_name,
                        'Rating': eval.rating,
                        'Score': eval.score,
                        'Max Score': eval.max_score,
                        'Weight (%)': eval.weight,
                        'Normalized Score (Score/Max)': normalized_score,
                        'Effective Weight (%) (Used)': effective_weight_pct,
                        'Contribution to Overall (%)': contribution_pct,
                        'Evidence': evidence_str,
                        'Evidence Timeframe': evidence_timeframe,
                        'AI Remarks': getattr(eval, "reasoning", "") or ""
                    })
    
    detailed_df = pd.DataFrame(detailed_rows)

    def _safe_sheet_name(name: str, used: set) -> str:
        """Excel sheet names must be <=31 chars and cannot contain: : \\ / ? * [ ]"""
        base = os.path.splitext(str(name or "File"))[0]
        for ch in [":", "\\", "/", "?", "*", "[", "]"]:
            base = base.replace(ch, "_")
        base = base.strip() or "File"
        base = base[:31]

        candidate = base
        i = 1
        while candidate in used or candidate.lower() == "history":
            suffix = f"_{i}"
            candidate = (base[: max(0, 31 - len(suffix))] + suffix).rstrip()
            i += 1
        used.add(candidate)
        return candidate
    
    # Write to Excel with multiple sheets:
    # - Summary
    # - One sheet per file (detailed evaluation rows for that file)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        used_names = {"Summary"}

        # Preserve the original batch ordering
        for r in results:
            file_name = r.get('filename') or "File"
            sheet_name = _safe_sheet_name(file_name, used_names)

            if r.get('evaluation_report'):
                if not detailed_df.empty and 'File Name' in detailed_df.columns:
                    df_file = detailed_df[detailed_df['File Name'] == file_name].copy()
                    if 'File Name' in df_file.columns:
                        df_file = df_file.drop(columns=['File Name'])
                else:
                    df_file = pd.DataFrame([])

                if df_file.empty:
                    df_file = pd.DataFrame([{"Status": "No detailed rows available"}])

                df_file.to_excel(writer, index=False, sheet_name=sheet_name)
            else:
                err_df = pd.DataFrame(
                    [
                        {
                            "Status": "Error",
                            "Error": r.get("error", ""),
                            "S3 Key": r.get("key", ""),
                        }
                    ]
                )
                err_df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    
    return output


def generate_tableau_dataset_excel(report) -> io.BytesIO:
    """
    Generate an Excel workbook for Tableau dashboards.

    Goal: include all datapoints used in the UI, as clean tables (one sheet per table):
    - CallSummary: single-row call-level metrics
    - Criteria: one row per criterion evaluation (includes call-level join keys)
    - CallIssues: one row per detected customer issue
    - ArticulationCoaching: one row per evaluation category coaching text
    - TranscriptSegments: one row per speaker segment (if available)
    - HoldSegments: one row per detected hold/silence segment (if available)
    - ModelCostComparison: cost comparison table (if cost inputs available)
    """
    import pandas as pd

    def _excel_safe_text(text: str, max_len: int = 32000) -> str:
        t = str(text or "")
        if len(t) <= max_len:
            return t
        return (t[: max_len - 1] + "…").strip()

    # Pull UI datapoints from session state (same sources the UI uses)
    transcript_text = st.session_state.get("transcript", "") or ""
    transcript_segments = st.session_state.get("transcript_segments") or []
    transcript_timing = st.session_state.get("transcript_timing", {}) or {}
    hold_summary = st.session_state.get("hold_summary", {}) or {}
    hold_analysis = st.session_state.get("hold_analysis", []) or []
    cso_wpm = st.session_state.get("cso_wpm")
    audio_duration_seconds = st.session_state.get("audio_duration_seconds")
    audio_duration_display = st.session_state.get("audio_duration", "--:--")

    # Cost breakdown (same logic as UI)
    cost_breakdown = None
    try:
        if audio_duration_seconds and transcript_text:
            cost_breakdown = calculate_total_processing_cost(float(audio_duration_seconds), transcript_text)
    except Exception:
        cost_breakdown = None

    compliance_status = "Passed" if float(getattr(report, "percentage_score", 0.0) or 0.0) >= 85 else "Review Needed"

    detected_name = getattr(report, "detected_agent_name", "") or ""
    matched_name = getattr(report, "detected_agent_canonical_name", "") or ""
    skills = getattr(report, "detected_agent_skills", []) or []
    skills_str = ", ".join([str(s) for s in skills if str(s).strip()])
    match_score = float(getattr(report, "detected_agent_match_score", 0.0) or 0.0)

    call_cls = getattr(report, "call_classification", "") or ""
    call_subject = getattr(report, "call_subject", "") or ""
    call_issues = getattr(report, "call_issues", []) or []
    if not isinstance(call_issues, list):
        call_issues = [str(call_issues)]

    # Summary row
    summary_row = {
        "Transcript ID": getattr(report, "transcript_id", "") or "",
        "Evaluation Timestamp": getattr(report, "evaluation_timestamp", "") or "",
        "Overall Score": float(getattr(report, "total_score", 0.0) or 0.0),
        "Overall Max Score": float(getattr(report, "max_possible_score", 0.0) or 0.0),
        "Overall %": float(getattr(report, "percentage_score", 0.0) or 0.0),
        "Overall Rating": getattr(report, "overall_rating", "") or "",
        "Compliance Status": compliance_status,
        "Raw Total Score": float(getattr(report, "raw_total_score", 0.0) or 0.0),
        "Raw Max Score": float(getattr(report, "raw_max_possible_score", 0.0) or 0.0),
        "Raw %": float(getattr(report, "raw_percentage_score", 0.0) or 0.0),
        "Weight Sum (used)": float(getattr(report, "weight_sum", 0.0) or 0.0),
        "Detected Agent Name": detected_name,
        "Matched Agent Name": matched_name,
        "Agent Skills (LI/GI/HI)": skills_str,
        "Name Match Score": match_score,
        "Call Classification": call_cls,
        "Call Subject": call_subject,
        "Customer Issues (pipe-delimited)": " | ".join([str(x).strip() for x in call_issues if str(x).strip()]),
        "Call Duration Display": audio_duration_display,
        "Call Start (sec)": float(transcript_timing.get("start_time", 0.0) or 0.0) if transcript_timing else 0.0,
        "Call End (sec)": float(transcript_timing.get("end_time", 0.0) or 0.0) if transcript_timing else 0.0,
        "Call Duration (sec)": float(transcript_timing.get("duration", 0.0) or 0.0) if transcript_timing else 0.0,
        "Hold Music (sec)": float(hold_summary.get("hold_music", 0.0) or 0.0),
        "Dead Silence (sec)": float(hold_summary.get("dead_silence", 0.0) or 0.0),
        "CSO WPM": float(cso_wpm) if cso_wpm is not None else "",
        "Transcript Char Count": len(transcript_text),
        "Transcript (truncated)": _excel_safe_text(transcript_text),
    }

    if cost_breakdown:
        trans = cost_breakdown.get("transcribe", {}) or {}
        bed = cost_breakdown.get("bedrock", {}) or {}
        summary_row.update(
            {
                "Transcribe Minutes": trans.get("duration_minutes", ""),
                "Transcribe Cost ($)": trans.get("cost", ""),
                "Transcribe Cost (formatted)": trans.get("cost_formatted", ""),
                "Bedrock Input Tokens": bed.get("input_tokens", ""),
                "Bedrock Output Tokens": bed.get("output_tokens", ""),
                "Bedrock Total Tokens": bed.get("total_tokens", ""),
                "Bedrock Cost ($)": bed.get("total_cost", ""),
                "Bedrock Cost (formatted)": bed.get("cost_formatted", ""),
                "Total Cost ($)": cost_breakdown.get("total_cost", ""),
                "Total Cost (formatted)": cost_breakdown.get("total_formatted", ""),
            }
        )

    call_summary_df = pd.DataFrame([summary_row])

    # Helper: timeframe lookup for evidence quotes
    def _normalize_tokens(t: str) -> str:
        if not t:
            return ""
        t = str(t).lower()
        return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t).split())

    def _find_timeframe_for_evidence(ev_text: str) -> str:
        if not ev_text or not transcript_segments:
            return ""
        ev_norm = _normalize_tokens(ev_text)
        if not ev_norm:
            return ""
        for seg in transcript_segments or []:
            seg_text = seg.get("text", "")
            seg_norm = _normalize_tokens(seg_text)
            if ev_norm and seg_norm and ev_norm in seg_norm:
                try:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start))
                    return format_time_range(start, end)
                except Exception:
                    return ""
        return ""

    # Criteria sheet
    criteria_rows = []
    weight_sum = float(getattr(report, "weight_sum", 0.0) or 0.0)
    for _agent_key, res in (getattr(report, "agent_results", {}) or {}).items():
        agent_category = getattr(res, "agent_name", _agent_key) or _agent_key
        for ev in getattr(res, "evaluations", []) or []:
            try:
                w = float(getattr(ev, "weight", 0.0) or 0.0)
                s = float(getattr(ev, "score", 0.0) or 0.0)
                m = float(getattr(ev, "max_score", 0.0) or 0.0)
            except Exception:
                w, s, m = 0.0, 0.0, 0.0

            normalized = (s / m) if (m and m != 0.0) else ""
            eff_w = (w / weight_sum * 100.0) if (weight_sum and w > 0.0 and m) else ""
            contribution = (eff_w * normalized) if (eff_w != "" and normalized != "") else ""

            evidence_list = list(getattr(ev, "evidence", None) or [])
            evidence_str = " | ".join([str(x) for x in evidence_list if str(x).strip()])
            first_ev = evidence_list[0] if evidence_list else ""
            timeframe = _find_timeframe_for_evidence(first_ev) if first_ev else ""

            criteria_rows.append(
                {
                    # Call-level join keys
                    "Transcript ID": getattr(report, "transcript_id", "") or "",
                    "Evaluation Timestamp": getattr(report, "evaluation_timestamp", "") or "",
                    "Matched Agent Name": matched_name,
                    "Call Classification": call_cls,
                    # Criterion-level fields
                    "Agent Category": agent_category,
                    "Criteria ID": (str(getattr(ev, "criteria_id", "") or "").split(".", 1)[1] if "." in str(getattr(ev, "criteria_id", "") or "") else (getattr(ev, "criteria_id", "") or "")),
                    "Criteria Name": getattr(ev, "criteria_name", "") or "",
                    "Rating": getattr(ev, "rating", "") or "",
                    "Weight (%)": w,
                    "Score": s,
                    "Max Score": m,
                    "Normalized Score (Score/Max)": normalized,
                    "Effective Weight (%) (Used)": eff_w,
                    "Contribution to Overall (%)": contribution,
                    "Evidence": evidence_str,
                    "Evidence Timeframe": timeframe,
                    "AI Remarks": getattr(ev, "reasoning", "") or "",
                }
            )

    criteria_df = pd.DataFrame(criteria_rows)

    # Call issues
    issues_rows = []
    for idx, issue in enumerate([str(x).strip() for x in call_issues if str(x).strip()], start=1):
        issues_rows.append(
            {
                "Transcript ID": getattr(report, "transcript_id", "") or "",
                "Evaluation Timestamp": getattr(report, "evaluation_timestamp", "") or "",
                "Issue #": idx,
                "Issue": issue,
            }
        )
    issues_df = pd.DataFrame(issues_rows)

    # Articulation coaching
    art = getattr(report, "articulation_suggestions", {}) or {}
    if not isinstance(art, dict):
        art = {}
    label_map = {
        "opening_greeting": "Opening Greeting",
        "verification": "Verification",
        "soft_skills": "Soft Skills",
        "enquiry_resolution": "Enquiry Resolution",
        "cross_selling": "Cross Selling",
        "wrap_up": "Wrap Up",
    }
    art_rows = []
    for key in ["opening_greeting", "verification", "soft_skills", "enquiry_resolution", "cross_selling", "wrap_up"]:
        txt = str(art.get(key, "") or "").strip()
        art_rows.append(
            {
                "Transcript ID": getattr(report, "transcript_id", "") or "",
                "Evaluation Timestamp": getattr(report, "evaluation_timestamp", "") or "",
                "Category Key": key,
                "Category": label_map.get(key, key),
                "Suggestion": txt,
            }
        )
    art_df = pd.DataFrame(art_rows)

    # Transcript segments (if available)
    seg_rows = []
    for seg in transcript_segments or []:
        seg_rows.append(
            {
                "Transcript ID": getattr(report, "transcript_id", "") or "",
                "Speaker": seg.get("speaker", ""),
                "Raw Speaker": seg.get("raw_speaker", ""),
                "Start (sec)": seg.get("start", ""),
                "End (sec)": seg.get("end", ""),
                "Text": _excel_safe_text(seg.get("text", "")),
            }
        )
    seg_df = pd.DataFrame(seg_rows)

    # Hold segments (if available)
    hold_rows = []
    for h in hold_analysis or []:
        hold_rows.append(
            {
                "Transcript ID": getattr(report, "transcript_id", "") or "",
                "Type": h.get("type", ""),
                "Start": h.get("start", ""),
                "End": h.get("end", ""),
                "Duration (sec)": h.get("duration", ""),
                "RMS": h.get("rms", ""),
            }
        )
    hold_df = pd.DataFrame(hold_rows)

    # Model cost comparison (if cost inputs available)
    model_cost_df = pd.DataFrame([])
    try:
        if cost_breakdown and isinstance(cost_breakdown, dict):
            bed = cost_breakdown.get("bedrock", {}) or {}
            input_tokens = int(bed.get("input_tokens", 0) or 0)
            output_tokens = int(bed.get("output_tokens", 0) or 0)
            comparison_rows = []
            for model_name, prices in (MODEL_PRICING_USD_PER_1K or {}).items():
                in_price = float(prices.get("input", 0.0))
                out_price = float(prices.get("output", 0.0))
                in_cost = (input_tokens / 1000.0) * in_price
                out_cost = (output_tokens / 1000.0) * out_price
                comparison_rows.append(
                    {
                        "Transcript ID": getattr(report, "transcript_id", "") or "",
                        "Model": model_name,
                        "Input $/1K": in_price,
                        "Output $/1K": out_price,
                        "Est. LLM Cost ($)": round(in_cost + out_cost, 6),
                    }
                )
            comparison_rows.sort(key=lambda r: r.get("Est. LLM Cost ($)", 0.0))
            model_cost_df = pd.DataFrame(comparison_rows)
    except Exception:
        model_cost_df = pd.DataFrame([])

    # Write workbook
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        call_summary_df.to_excel(writer, index=False, sheet_name="CallSummary")
        criteria_df.to_excel(writer, index=False, sheet_name="Criteria")
        issues_df.to_excel(writer, index=False, sheet_name="CallIssues")
        art_df.to_excel(writer, index=False, sheet_name="ArticulationCoaching")

        # Optional sheets (can be empty)
        seg_df.to_excel(writer, index=False, sheet_name="TranscriptSegments")
        hold_df.to_excel(writer, index=False, sheet_name="HoldSegments")
        if not model_cost_df.empty:
            model_cost_df.to_excel(writer, index=False, sheet_name="ModelCostComparison")

    output.seek(0)
    return output


def calculate_transcribe_cost(duration_seconds: float) -> dict:
    """
    Calculate AWS Transcribe cost based on audio duration.
    
    Args:
        duration_seconds: Audio duration in seconds
    
    Returns:
        dict with minutes, cost, and formatted strings
    """
    # Round up to nearest second, minimum 15 seconds
    duration_seconds = max(15, duration_seconds)
    duration_minutes = duration_seconds / 60
    cost = duration_minutes * TRANSCRIBE_PRICE_PER_MINUTE
    
    return {
        'duration_seconds': duration_seconds,
        'duration_minutes': round(duration_minutes, 2),
        'cost': round(cost, 4),
        'cost_formatted': f"${cost:.4f}"
    }


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text (approximate: ~4 chars per token for English).
    This is a rough estimate - actual tokenization varies by model.
    """
    if not text:
        return 0
    # Approximate: 1 token ≈ 4 characters for English text
    return max(1, len(text) // 4)


def calculate_bedrock_cost(input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate AWS Bedrock Claude 3.5 Sonnet cost based on token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        dict with token counts, costs, and formatted strings
    """
    input_cost = (input_tokens / 1000) * BEDROCK_INPUT_PRICE_PER_1K
    output_cost = (output_tokens / 1000) * BEDROCK_OUTPUT_PRICE_PER_1K
    total_cost = input_cost + output_cost
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'input_cost': round(input_cost, 6),
        'output_cost': round(output_cost, 6),
        'total_cost': round(total_cost, 6),
        'cost_formatted': f"${total_cost:.4f}"
    }


def estimate_evaluation_cost(transcript_text: str, num_agents: int = 6) -> dict:
    """
    Estimate the total Bedrock cost for running quality evaluation.
    
    Each agent receives the transcript as input and produces evaluation output.
    
    Args:
        transcript_text: The transcript being evaluated
        num_agents: Number of evaluation agents (default 6)
    
    Returns:
        dict with estimated costs
    """
    # Input tokens: transcript + system prompt (~2000 tokens) per agent
    transcript_tokens = estimate_token_count(transcript_text)
    system_prompt_tokens = 2000  # Approximate system prompt size
    input_tokens_per_agent = transcript_tokens + system_prompt_tokens
    total_input_tokens = input_tokens_per_agent * num_agents
    
    # Output tokens: estimated ~800 tokens per agent for evaluation response
    output_tokens_per_agent = 800
    total_output_tokens = output_tokens_per_agent * num_agents
    
    return calculate_bedrock_cost(total_input_tokens, total_output_tokens)


def calculate_total_processing_cost(duration_seconds: float, transcript_text: str) -> dict:
    """
    Calculate total cost for processing one audio file (transcription + evaluation).
    
    Args:
        duration_seconds: Audio duration in seconds
        transcript_text: The transcript text
    
    Returns:
        dict with all cost breakdowns
    """
    transcribe_cost = calculate_transcribe_cost(duration_seconds)
    bedrock_cost = estimate_evaluation_cost(transcript_text)
    
    total_cost = transcribe_cost['cost'] + bedrock_cost['total_cost']
    
    return {
        'transcribe': transcribe_cost,
        'bedrock': bedrock_cost,
        'total_cost': round(total_cost, 4),
        'total_formatted': f"${total_cost:.4f}"
    }


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_time_range(start: float, end: float) -> str:
    """Format a start/end time range for display."""
    return f"{format_timestamp(start)} → {format_timestamp(end)}"


def parse_transcript(response_json, use_channel_id=False):
    """
    Parse transcript from AWS Transcribe response.
    
    Supports both:
    - Speaker Diarization mode (mono audio)
    - Channel Identification mode (stereo audio with agent/customer on separate channels)
    
    Args:
        response_json: AWS Transcribe JSON response
        use_channel_id: If True, parse channel identification results
    
    Returns:
        Tuple of (transcript_text, segments_info, timing_info)
    """
    
    # Handle Channel Identification mode (stereo audio)
    if use_channel_id and 'results' in response_json and 'channel_labels' in response_json['results']:
        logger.info("Parsing Channel Identification results")
        channel_labels = response_json['results']['channel_labels']
        channels = channel_labels.get('channels', [])
        
        all_segments = []
        call_start = None
        call_end = None
        
        for channel in channels:
            channel_label = channel.get('channel_label', '')
            items = channel.get('items', [])
            
            # Map channel to speaker role:
            # ch_0 (left channel) = Agent/CSO
            # ch_1 (right channel) = Customer
            if channel_label == 'ch_0':
                speaker_role = 'CSO'
            elif channel_label == 'ch_1':
                speaker_role = 'Customer'
            else:
                speaker_role = f'Speaker_{channel_label}'
            
            # Group consecutive items into segments
            current_segment = []
            segment_start = None
            segment_end = None
            
            for item in items:
                if item.get('type') == 'pronunciation':
                    if segment_start is None:
                        segment_start = float(item.get('start_time', 0))
                    segment_end = float(item.get('end_time', segment_start))
                    current_segment.append(item['alternatives'][0]['content'])
                elif item.get('type') == 'punctuation':
                    current_segment.append(item['alternatives'][0]['content'])
                
                # Create segment on long pause or end
                if current_segment and segment_end:
                    call_start = segment_start if call_start is None else min(call_start, segment_start)
                    call_end = segment_end if call_end is None else max(call_end, segment_end)
            
            # Final segment for this channel
            if current_segment and segment_start is not None:
                text = " ".join(current_segment)
                text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
                
                all_segments.append({
                    "speaker": speaker_role,
                    "raw_speaker": channel_label,
                    "start": segment_start,
                    "end": segment_end or segment_start,
                    "text": text
                })
        
        # Sort all segments by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Build transcript lines
        transcript_lines = []
        for seg in all_segments:
            timestamp = format_timestamp(seg['start'])
            transcript_lines.append(f"{timestamp} **{seg['speaker']}**: {seg['text']}")
        
        timing_info = None
        if call_start is not None and call_end is not None:
            timing_info = {
                "start_time": call_start,
                "end_time": call_end,
                "duration": max(0.0, call_end - call_start)
            }
        
        return "\n\n".join(transcript_lines), all_segments, timing_info
    
    # Handle Speaker Diarization mode (mono audio)
    elif 'results' in response_json and 'speaker_labels' in response_json['results']:
        logger.info("Parsing Speaker Diarization results")
        speaker_labels = response_json['results']['speaker_labels']
        items = response_json['results']['items']
        
        segments = speaker_labels['segments']
        transcript_lines = []
        segments_info = []
        call_start = None
        call_end = None
        
        item_index = 0
        tmp_segments = []
        for segment in segments:
            speaker = segment['speaker_label']
            start_time = float(segment['start_time'])
            end_time = float(segment['end_time'])
            call_start = start_time if call_start is None else min(call_start, start_time)
            call_end = end_time if call_end is None else max(call_end, end_time)
            
            segment_text = []
            
            while item_index < len(items):
                item = items[item_index]
                
                if item['type'] == 'punctuation':
                    segment_text.append(item['alternatives'][0]['content'])
                    item_index += 1
                    continue
                    
                if 'start_time' in item:
                    item_start = float(item['start_time'])
                    if item_start < end_time:
                        segment_text.append(item['alternatives'][0]['content'])
                        item_index += 1
                    else:
                        break
                else:
                    item_index += 1

            text = " ".join(segment_text)
            text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            tmp_segments.append(
                {
                    "raw_speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                }
            )

        def _infer_cso_raw_speaker(segs):
            """
            Infer which diarized speaker label (e.g., spk_0/spk_1) is the CSO/Agent.
            AWS diarization labels are arbitrary; do NOT assume spk_0 is always the agent.
            """
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
                # Small preference for whoever appears in the opening turns
                score += 1
                scores[raw] = score

            if scores:
                return max(scores.items(), key=lambda kv: kv[1])[0]

            return segs[0].get("raw_speaker") or "spk_0"

        cso_raw = _infer_cso_raw_speaker(tmp_segments)
        logger.info(f"Inferred CSO diarization label: {cso_raw}")

        for seg in tmp_segments:
            speaker = seg.get("raw_speaker", "")
            start_time = float(seg.get("start", 0.0))
            end_time = float(seg.get("end", start_time))
            text = seg.get("text", "") or ""

            speaker_role = "CSO" if speaker == cso_raw else "Customer"
            timestamp = format_timestamp(start_time)

            transcript_lines.append(f"{timestamp} **{speaker_role}**: {text}")
            segments_info.append(
                {
                    "speaker": speaker_role,
                    "raw_speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                }
            )
            
        timing_info = None
        if call_start is not None and call_end is not None:
            timing_info = {
                "start_time": call_start,
                "end_time": call_end,
                "duration": max(0.0, call_end - call_start)
            }
            
        return "\n\n".join(transcript_lines), segments_info, timing_info
    
    # Fallback: simple transcript without speaker labels
    else:
        logger.info("Parsing simple transcript (no speaker labels)")
        text = response_json['results']['transcripts'][0]['transcript']
        return text, [], None


def transcribe_audio(
    s3_uri,
    job_name,
    transcribe_client,
    max_speakers=2,
    use_channel_id=False,
    language_code='en-US',
    auto_detect_language=True,
    identify_multiple_languages=False,
    custom_vocabulary=None,
    media_format: str = "wav",
):
    """
    Transcribe audio using AWS Transcribe with enhanced settings.
    
    Returns:
        Tuple of (transcript_text, segments_info, timing_info, language_info) or None
    """
    from language_pipeline import build_transcription_language_params, extract_language_metadata

    logger.info(
        "Starting Transcription job: JobName=%s, URI=%s, MaxSpeakers=%s, ChannelID=%s, "
        "AutoDetect=%s, MultiLang=%s, FixedLanguage=%s",
        job_name,
        s3_uri,
        max_speakers,
        use_channel_id,
        auto_detect_language,
        identify_multiple_languages,
        language_code,
    )
    
    try:
        settings = {}
        
        if use_channel_id:
            settings['ChannelIdentification'] = True
            logger.info("Using Channel Identification mode (stereo audio)")
        else:
            settings['ShowSpeakerLabels'] = True
            settings['MaxSpeakerLabels'] = max_speakers
            logger.info(f"Using Speaker Diarization mode with max {max_speakers} speakers")
        
        if custom_vocabulary and custom_vocabulary.strip():
            settings['VocabularyName'] = custom_vocabulary.strip()
            logger.info(f"Using custom vocabulary: {custom_vocabulary}")

        language_params = build_transcription_language_params(
            language_code=language_code,
            auto_detect_language=auto_detect_language,
            identify_multiple_languages=identify_multiple_languages,
        )

        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat=media_format,
            Settings=settings,
            **language_params,
        )

        with st.spinner("✨ Transcribing audio... This may take a few minutes for longer files."):
            while True:
                status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                logger.info(f"Transcription status: {job_status}")
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(2)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            logger.info(f"Transcription completed. Fetching result from {transcript_uri}")
            response = requests.get(transcript_uri, verify=False)
            transcript_json = response.json()
            language_info = extract_language_metadata(status['TranscriptionJob'], transcript_json)
            parsed = parse_transcript(transcript_json, use_channel_id=use_channel_id)
            return (*parsed, language_info)
        else:
            failure_reason = status.get('TranscriptionJob', {}).get('FailureReason', 'Unknown reason')
            logger.error(f"Transcription failed. Reason: {failure_reason}")
            st.error(f"Transcription failed: {failure_reason}")
            return None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        st.error(f"Error during transcription: {e}")
        return None


def plot_waveform(data, sr, title):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(data, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    return fig


def process_audio(uploaded_file):
    """Load and process audio file"""
    try:
        # Streamlit's uploaded file can be partially consumed by st.audio; load from bytes for stability.
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        y, sr = _load_audio_from_bytes(
            file_bytes,
            mono=True,
            suffix=os.path.splitext(getattr(uploaded_file, "name", "") or "")[1].lower() or ".wav",
        )
        # Calculate duration in minutes and seconds
        duration_sec = librosa.get_duration(y=y, sr=sr)
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"
        st.session_state['audio_duration'] = duration_str
        st.session_state['audio_duration_seconds'] = duration_sec
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None
    return y, sr


def detect_hold_segments(segments, audio_data, sr, min_gap=10.0, silence_threshold=0.0015):
    """Detect hold segments and classify as music vs dead silence"""
    if not segments or audio_data is None or sr is None:
        return []

    holds = []
    sorted_segments = sorted(segments, key=lambda x: x['start'])

    gaps = []
    prev_end = 0.0

    for seg in sorted_segments:
        gap_start = prev_end
        gap_end = seg['start']
        if gap_end - gap_start >= min_gap:
            gaps.append((gap_start, gap_end))
        prev_end = max(prev_end, seg['end'])

    total_duration = st.session_state.get('audio_duration_seconds')
    if total_duration and prev_end < total_duration and (total_duration - prev_end) >= min_gap:
        gaps.append((prev_end, total_duration))

    for gap_start, gap_end in gaps:
        start_idx = max(0, int(gap_start * sr))
        end_idx = min(len(audio_data), int(gap_end * sr))
        snippet = audio_data[start_idx:end_idx]
        if snippet.size == 0:
            continue
        rms = float(np.sqrt(np.mean(snippet**2)))
        classification = "Hold Music/Audio" if rms >= silence_threshold else "Dead Silence"
        holds.append({
            "start": format_timestamp(gap_start),
            "end": format_timestamp(gap_end),
            "duration": gap_end - gap_start,
            "type": classification,
            "rms": rms
        })

    return holds


def summarize_hold_segments(holds):
    """Summarize total durations for hold music and dead silence segments"""
    summary = {
        "hold_music": 0.0,
        "dead_silence": 0.0
    }
    for hold in holds:
        duration = float(hold.get("duration", 0.0))
        if hold.get("type") == "Hold Music/Audio":
            summary["hold_music"] += duration
        elif hold.get("type") == "Dead Silence":
            summary["dead_silence"] += duration
    return summary


def compute_timing_from_segments(segments, audio_duration=None):
    """Compute start/end/duration from segments or fall back to total audio duration"""
    if segments:
        start_time = min(float(s.get("start", 0.0)) for s in segments)
        end_time = max(float(s.get("end", 0.0)) for s in segments)
    elif audio_duration is not None:
        start_time = 0.0
        end_time = float(audio_duration)
    else:
        return {}

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration": max(0.0, end_time - start_time)
    }


def compute_words_per_minute(segments, timing_info=None, speaker_role="CSO"):
    """
    Calculate words per minute for a specific speaker.
    Uses actual speaking time when available; falls back to overall duration.
    """
    if not segments:
        return None

    speech_segments = [s for s in segments if s.get("speaker") == speaker_role]
    if not speech_segments:
        return None

    total_words = sum(len(s.get("text", "").split()) for s in speech_segments)
    speech_time = sum(
        max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        for s in speech_segments
    )

    base_duration = timing_info.get("duration") if timing_info else None
    duration_seconds = speech_time if speech_time > 0 else base_duration

    if not duration_seconds or duration_seconds <= 0:
        return None

    return round(total_words / (duration_seconds / 60.0), 1)


def find_evidence_timestamp(evidence_text, segments):
    """Locate the start time of the segment that contains the evidence text."""
    if not evidence_text or not segments:
        return None
    evidence_lower = evidence_text.lower()
    for seg in segments:
        seg_text = seg.get("text", "")
        if evidence_lower in seg_text.lower():
            try:
                return float(seg.get("start", 0.0))
            except Exception:
                return None
    return None


def create_audio_snippet(start_sec, duration_sec=15.0):
    """
    Create a short audio snippet buffer starting at start_sec for playback.
    Returns BytesIO or None if unavailable.
    """
    y = st.session_state.get('reduced_noise_y')
    sr = st.session_state.get('sr')
    if y is None or sr is None:
        return None
    if start_sec is None:
        return None

    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(y), int((start_sec + duration_sec) * sr))
    if end_idx <= start_idx:
        return None

    snippet = y[start_idx:end_idx]
    buffer = io.BytesIO()
    sf.write(buffer, snippet, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer


def run_quality_evaluation(transcript_text, aws_access_key_id, aws_secret_access_key, aws_region, translate_to_english=True, detected_languages=None):
    """Run the multi-agent quality evaluation on the transcript"""
    try:
        aws_access_key_id = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
        aws_secret_access_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
        aws_region = aws_region or DEFAULT_AWS_REGION

        with st.spinner("✨ Translating (if needed) and evaluating the transcript..."):
            report, _prepared, _orchestrator = _evaluate_transcript_with_translation(
                transcript_text=transcript_text,
                segments=st.session_state.get("transcript_segments") or [],
                filename="manual_transcript",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
                run_parallel=run_parallel,
                translate_to_english=translate_to_english,
                detected_languages=detected_languages or st.session_state.get("detected_languages") or [],
            )

        return report
        
    except ImportError as e:
        st.error(f"Error importing evaluation modules: {e}")
        st.info("Please ensure all required packages are installed: `pip install langchain langchain-aws`")
        return None
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        st.error(f"Error during evaluation: {e}")
        return None


def _timestamp_for_filename(iso_timestamp: str) -> str:
    """Convert ISO timestamp to a filesystem-friendly timestamp."""
    try:
        # handle trailing 'Z' if present
        cleaned = (iso_timestamp or "").replace("Z", "")
        dt = datetime.fromisoformat(cleaned)
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_s3_key_part(text: str) -> str:
    """Sanitize a string for use in S3 object keys / filenames."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    out = []
    for ch in str(text or ""):
        out.append(ch if ch in allowed else "_")
    safe = "".join(out).strip("_")
    return safe or "report"


def _maybe_upload_excel_bytes_to_s3(
    *,
    excel_bytes: bytes,
    bucket: str,
    object_key: str,
    s3_client=None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region: str | None = None,
) -> bool:
    """Upload excel bytes to S3 once per session (idempotent by object_key)."""
    if not bucket or not object_key or not excel_bytes:
        return False

    uploaded = st.session_state.setdefault("uploaded_excel_keys", set())
    if object_key in uploaded:
        return True

    try:
        if s3_client is None:
            session = create_aws_session(
                region_name=aws_region or DEFAULT_AWS_REGION,
                aws_access_key_id=aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID,
                aws_secret_access_key=aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY,
            )
            s3_client = session.client("s3", verify=False)

        buf = io.BytesIO(excel_bytes)
        buf.seek(0)
        s3_client.upload_fileobj(buf, bucket, object_key)
        uploaded.add(object_key)
        st.session_state["uploaded_excel_keys"] = uploaded
        return True
    except Exception as e:
        logger.error(f"Error uploading Excel to s3://{bucket}/{object_key}: {e}")
        return False


def _safe_session_key(text: str) -> str:
    """Make a string safe to use as a Streamlit widget/session key."""
    return "".join(ch if str(ch).isalnum() else "_" for ch in str(text))


def _get_rating_options_for_criteria(criteria_id: str) -> list:
    """Return valid rating options for a criterion, in a consistent order."""
    try:
        from evaluation_criteria import get_criteria_by_path
    except Exception:
        return ["Desirable", "Expected", "Basic", "Undesirable"]

    criteria = get_criteria_by_path(criteria_id) or {}
    ratings = list((criteria.get("ratings") or {}).keys())
    preferred_order = ["Desirable", "Expected", "Basic", "Undesirable"]
    ordered = [r for r in preferred_order if r in ratings]
    return ordered or preferred_order


def _score_for_rating(criteria_id: str, rating: str, fallback_score: float) -> float:
    """Map (criteria_id, rating) -> score using evaluation criteria definitions."""
    try:
        from evaluation_criteria import get_criteria_by_path
    except Exception:
        return fallback_score

    criteria = get_criteria_by_path(criteria_id) or {}
    rating_info = (criteria.get("ratings") or {}).get(rating) or {}
    try:
        return float(rating_info.get("score", fallback_score))
    except Exception:
        return fallback_score


def _ensure_override_state_initialized(report) -> tuple[str, dict, dict]:
    """
    Ensure baseline and overrides are initialized for a given report.
    Returns (report_key, baseline_map, overrides_map_for_report).
    """
    report_key = getattr(report, "transcript_id", None) or "default_report"

    override_state = st.session_state.setdefault(
        "override_state",
        {"baseline_by_report": {}, "overrides_by_report": {}},
    )

    baseline_by_report = override_state.setdefault("baseline_by_report", {})
    overrides_by_report = override_state.setdefault("overrides_by_report", {})

    if report_key not in baseline_by_report:
        baseline = {}
        try:
            for agent_result in (getattr(report, "agent_results", {}) or {}).values():
                for ev in getattr(agent_result, "evaluations", []) or []:
                    baseline[ev.criteria_id] = {"rating": ev.rating, "score": ev.score}
        except Exception:
            baseline = {}
        baseline_by_report[report_key] = baseline

    overrides = overrides_by_report.setdefault(report_key, {})
    return report_key, baseline_by_report[report_key], overrides


def _on_override_change(report_key: str, criteria_id: str, widget_key: str):
    """Widget callback: persist overrides only when they differ from AI baseline."""
    override_state = st.session_state.get("override_state", {})
    baseline_by_report = override_state.get("baseline_by_report", {})
    overrides_by_report = override_state.get("overrides_by_report", {})

    baseline = baseline_by_report.get(report_key, {}) or {}
    ai_rating = (baseline.get(criteria_id) or {}).get("rating")
    new_rating = st.session_state.get(widget_key)

    overrides = overrides_by_report.setdefault(report_key, {})
    if ai_rating is not None and new_rating == ai_rating:
        overrides.pop(criteria_id, None)
    else:
        overrides[criteria_id] = new_rating

    override_state["overrides_by_report"] = overrides_by_report
    st.session_state["override_state"] = override_state


def _apply_overrides_to_report(report, overrides: dict):
    """Return a report with overrides applied (mutates the provided report instance)."""
    if not overrides:
        return report

    for agent_result in (getattr(report, "agent_results", {}) or {}).values():
        for ev in getattr(agent_result, "evaluations", []) or []:
            if ev.criteria_id not in overrides:
                continue
            new_rating = overrides[ev.criteria_id]
            ev.rating = new_rating
            ev.score = _score_for_rating(ev.criteria_id, new_rating, ev.score)
        try:
            agent_result.calculate_totals()
        except Exception:
            pass

    try:
        report.calculate_totals()
        report.identify_strengths_and_improvements()
        report.generate_summary()
    except Exception:
        pass

    return report


def display_evaluation_report(report):
    """Display the evaluation report in the new UI style"""
    from orchestrator import DetailedReportGenerator
    
    timing_info = st.session_state.get('transcript_timing', {}) or {}
    hold_summary = st.session_state.get('hold_summary', {}) or {}
    cso_wpm = st.session_state.get('cso_wpm')

    # --- Override AI (session-safe) ---
    report_key, _baseline_map, overrides = _ensure_override_state_initialized(report)
    effective_report = _apply_overrides_to_report(copy.deepcopy(report), overrides)
    report = effective_report

    if getattr(report, "detected_languages", None) or getattr(report, "was_translated", False):
        lang_text = ", ".join(getattr(report, "detected_languages", []) or []) or "Unknown"
        if getattr(report, "was_translated", False):
            st.info(f"🌐 Detected language(s): **{lang_text}**. Transcript translated to English for evaluation.")
        else:
            st.info(f"🌐 Detected language(s): **{lang_text}**.")

 
    st.markdown("###")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        
        if report.percentage_score >= 80:
            score_color = "#28a745" # Green
        elif report.percentage_score >= 60:
            score_color = "#2563EB" # Blue
        else:
            score_color = "#dc3545" # Red

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Score</div>
            <div class="metric-value score" style="color: {score_color}">{report.total_score:.1f} / {report.max_possible_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        agent_name = (
            getattr(report, "detected_agent_canonical_name", "")  # matched from roster if available
            or getattr(report, "detected_agent_name", "")
            or "Unknown"
        )
        agent_skills = getattr(report, "detected_agent_skills", []) or []
        skills_str = ", ".join([str(s) for s in agent_skills if str(s).strip()])
        match_score = float(getattr(report, "detected_agent_match_score", 0.0) or 0.0)
        roster_path = (os.getenv("AGENT_ROSTER_PATH", "") or "").strip()
        roster_file = os.path.basename(roster_path) if roster_path else ""
        sub_text = ""
        if getattr(report, "detected_agent_canonical_name", ""):
            sub_text = f"Skills: {skills_str or 'Unknown'} · Match: {match_score:.2f}"
        elif roster_path:
            sub_text = f"No roster match · Roster: {roster_file or roster_path}"
        else:
            sub_text = "Roster not configured"
        sub = f'<div class="metric-label" style="margin-top:0.35rem;">{sub_text}</div>'
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Agent</div>
            <div class="metric-value">{agent_name}</div>
            {sub}
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        compliance_status = "Passed" if report.percentage_score >= 85 else "Review Needed"
        color = "#28a745" if compliance_status == "Passed" else "#2563EB"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Compliance Status</div>
            <div class="metric-value" style="color: {color}">{compliance_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        duration = st.session_state.get('audio_duration', "--:--")
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Call Duration</div>
            <div class="metric-value">{duration}</div>
        </div>
        """, unsafe_allow_html=True)

   
    m5, m6, m7 = st.columns(3)

    start_ts = format_timestamp(timing_info.get('start_time', 0.0)) if timing_info else "--:--"
    end_ts = format_timestamp(timing_info.get('end_time', timing_info.get('duration', 0.0))) if timing_info else "--:--"
    duration_ts = format_timestamp(timing_info.get('duration', st.session_state.get('audio_duration_seconds', 0.0))) if timing_info else "--:--"

    with m5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Call Timing</div>
            <div class="metric-value">{start_ts} → {end_ts}</div>
            <div class="metric-label" style="margin-top:0.35rem;">Elapsed: {duration_ts}</div>
        </div>
        """, unsafe_allow_html=True)

    with m6:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Hold & Silence</div>
            <div class="metric-value">
                🎵 {hold_summary.get('hold_music', 0.0):.1f}s | 🔇 {hold_summary.get('dead_silence', 0.0):.1f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m7:
        wpm_display = f"{cso_wpm} WPM" if cso_wpm is not None else "N/A"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CSO Speech Rate</div>
            <div class="metric-value">{wpm_display}</div>
        </div>
        """, unsafe_allow_html=True)

    # Pricing metrics row
    st.markdown("### 💰 Cost Estimation")
    
    # Calculate costs
    audio_duration = st.session_state.get('audio_duration_seconds', 0)
    transcript = st.session_state.get('transcript', '')
    
    if audio_duration > 0 and transcript:
        cost_breakdown = calculate_total_processing_cost(audio_duration, transcript)
        
        p1, p2, p3, p4 = st.columns(4)
        
        with p1:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #17a2b8;">
                <div class="metric-label">Transcribe Cost</div>
                <div class="metric-value">{cost_breakdown['transcribe']['cost_formatted']}</div>
                <div class="metric-label" style="font-size: 0.75rem;">{cost_breakdown['transcribe']['duration_minutes']} min @ $0.024/min</div>
            </div>
            """, unsafe_allow_html=True)
        
        with p2:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #6f42c1;">
                <div class="metric-label">Bedrock Token Cost</div>
                <div class="metric-value">{cost_breakdown['bedrock']['cost_formatted']}</div>
                <div class="metric-label" style="font-size: 0.75rem;">{cost_breakdown['bedrock']['total_tokens']:,} tokens</div>
            </div>
            """, unsafe_allow_html=True)
        
        with p3:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #28a745;">
                <div class="metric-label">Total Cost</div>
                <div class="metric-value" style="color: #28a745;">{cost_breakdown['total_formatted']}</div>
                <div class="metric-label" style="font-size: 0.75rem;">Per evaluation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with p4:
            st.markdown(f"""
            <div class="metric-container" style="border-left-color: #2563EB;">
                <div class="metric-label">Token Breakdown</div>
                <div class="metric-value" style="font-size: 1rem;">In: {cost_breakdown['bedrock']['input_tokens']:,}</div>
                <div class="metric-label" style="font-size: 0.75rem;">Out: {cost_breakdown['bedrock']['output_tokens']:,}</div>
            </div>
            """, unsafe_allow_html=True)

        # Model comparison (cost only; no extra model calls)
        st.markdown("#### 🤖 Model Cost Comparison (Estimated)")
        input_tokens = int(cost_breakdown["bedrock"]["input_tokens"])
        output_tokens = int(cost_breakdown["bedrock"]["output_tokens"])
        total_tokens = input_tokens + output_tokens
        st.caption(f"Token estimate used for comparison: In={input_tokens:,}, Out={output_tokens:,}, Total={total_tokens:,}")

        comparison_rows = []
        for model_name, prices in MODEL_PRICING_USD_PER_1K.items():
            in_price = float(prices.get("input", 0.0))
            out_price = float(prices.get("output", 0.0))
            in_cost = (input_tokens / 1000.0) * in_price
            out_cost = (output_tokens / 1000.0) * out_price
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Input $/1K": in_price,
                    "Output $/1K": out_price,
                    "Est. LLM Cost ($)": round(in_cost + out_cost, 6),
                }
            )

        comparison_rows.sort(key=lambda r: r.get("Est. LLM Cost ($)", 0.0))
        try:
            st.dataframe(comparison_rows, hide_index=True, width="stretch")
        except TypeError:
            st.dataframe(comparison_rows, hide_index=True, use_container_width=True)
    else:
        st.info("💡 Cost estimation requires audio duration and transcript data.")

    # ------------------------------------------------------------------
    # Call intent + issues + articulation coaching
    # ------------------------------------------------------------------
    st.markdown("### 🧭 Call Intent & Issues")
    c1, c2 = st.columns([1, 2])
    with c1:
        call_cls = getattr(report, "call_classification", "") or "Unknown"
        st.markdown(f"""
        <div class="metric-container" style="border-left-color: #17a2b8;">
            <div class="metric-label">Call Classification</div>
            <div class="metric-value" style="font-size: 1.25rem;">{call_cls}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        subject = getattr(report, "call_subject", "") or ""
        issues = getattr(report, "call_issues", []) or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        issues = [str(x).strip() for x in issues if str(x).strip()]

        if subject:
            st.markdown(f"**Subject:** {subject}")
        if issues:
            st.markdown("**Customer issues raised:**")
            for it in issues[:10]:
                st.markdown(f"- {it}")
        if not subject and not issues:
            st.info("Call intent/issue classification not available (insights agent not run or failed).")

    st.markdown("### 🗣️ Articulation Coaching (by category)")
    suggestions = getattr(report, "articulation_suggestions", {}) or {}
    if not isinstance(suggestions, dict):
        suggestions = {}

    label_map = {
        "opening_greeting": "Opening Greeting",
        "verification": "Verification",
        "soft_skills": "Soft Skills",
        "enquiry_resolution": "Enquiry Resolution",
        "cross_selling": "Cross Selling",
        "wrap_up": "Wrap Up",
    }
    any_suggestion = any(str(v).strip() for v in (suggestions or {}).values())
    if not any_suggestion:
        st.info("No articulation suggestions available.")
    else:
        for key in ["opening_greeting", "verification", "soft_skills", "enquiry_resolution", "cross_selling", "wrap_up"]:
            txt = str(suggestions.get(key, "") or "").strip()
            if not txt:
                continue
            with st.expander(f"{label_map.get(key, key)}"):
                st.markdown(txt)

    st.markdown("---")

    

    col_left, col_right = st.columns([2, 1])
    
   
    with col_left:
        st.markdown('<div class="section-header">📝 Detailed Scorecard</div>', unsafe_allow_html=True)
        
        
        all_evaluations = []
        for agent_name, result in report.agent_results.items():
            all_evaluations.extend(result.evaluations)
            
      
        for eval in all_evaluations:
            
            
            expander_title = f"{eval.criteria_name} - {eval.score}/{eval.max_score} pts"
            if eval.score < eval.max_score:
                expander_title = "⚠️ " + expander_title
            else:
                expander_title = "✅ " + expander_title
                
            with st.expander(expander_title, expanded=False):
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.markdown("**AI Findings:**")
                    st.markdown(f"<div class='finding-box'>{eval.reasoning}</div>", unsafe_allow_html=True)
                    
                    if eval.evidence:
                        evidence_text = eval.evidence[0]
                        st.markdown(f"""
                        <div class="evidence-box">
                            💡 Evidence: "{evidence_text}"
                        </div>
                        """, unsafe_allow_html=True)

                        
                        evidence_start = find_evidence_timestamp(
                            evidence_text,
                            st.session_state.get("transcript_segments", [])
                        )
                        snippet_buffer = create_audio_snippet(evidence_start, duration_sec=15.0)

                        if snippet_buffer:
                            start_label = format_timestamp(evidence_start) if evidence_start is not None else "unknown"
                            st.markdown(f"**🎧 Play snippet (starts at {start_label})**")
                            st.audio(snippet_buffer, format='audio/wav')
                        else:
                            st.info("Audio snippet unavailable (no matching timestamp or audio not processed).")
                
                with c2:
                    current_rating = eval.rating
                    
                    st.markdown("**Rating:**")
                    # Display the current (possibly overridden) rating
                    st.markdown(f"<span class='rating-badge badge-{current_rating}'>{current_rating}</span>", unsafe_allow_html=True)
                    
                    st.markdown("<br>**Override AI:**", unsafe_allow_html=True)
                    
                    rating_options = _get_rating_options_for_criteria(eval.criteria_id)
                    current_index = rating_options.index(current_rating) if current_rating in rating_options else 0
                    widget_key = f"override_{_safe_session_key(report_key)}_{_safe_session_key(eval.criteria_id)}"
                    
                    new_rating = st.selectbox(
                        "Select Rating",
                        options=rating_options,
                        index=current_index,
                        key=widget_key,
                        on_change=_on_override_change,
                        kwargs={
                            "report_key": report_key,
                            "criteria_id": eval.criteria_id,
                            "widget_key": widget_key,
                        },
                        label_visibility="collapsed"
                    )
                    # Apply score override for immediate per-criterion display (totals are handled via effective_report)
                    if new_rating != current_rating:
                        eval.rating = new_rating
                        eval.score = _score_for_rating(eval.criteria_id, new_rating, eval.score)
                    
                    
    with col_right:
        st.markdown('<div class="section-header">🎧 Audio & Transcript</div>', unsafe_allow_html=True)
        
      
        if st.session_state.get('audio_cleaned'):
            try:
                y_clean = st.session_state.get('reduced_noise_y')
                sr = st.session_state.get('sr')
                if y_clean is not None:
                    buffer = io.BytesIO()
                    sf.write(buffer, y_clean, sr, format='WAV', subtype='PCM_16')
                    st.audio(buffer, format='audio/wav')
            except Exception as e:
                st.warning("Could not load audio player here.")
        
        st.markdown("###")
        
       
        st.markdown("**Transcript Search**")
        search_term = st.text_input("Find keyword", placeholder="Search transcript...", label_visibility="collapsed")
        
        
        st.markdown("**Live Transcript**")
        transcript_text = st.session_state.get('transcript', "No transcript available.")
        
       
        if search_term:
            filtered_lines = [line for line in transcript_text.split('\n') if search_term.lower() in line.lower()]
            display_text = "\n".join(filtered_lines) if filtered_lines else "No matches found."
        else:
            display_text = transcript_text
            
        st.text_area(
            "Transcript content",
            display_text,
            height=400,
            label_visibility="collapsed",
            key="transcript_right_col"
        )

        hold_info = st.session_state.get('hold_analysis', [])
        st.markdown("**Hold Detection**")
        if hold_info:
            for hold in hold_info:
                icon = "🎵" if hold["type"] == "Hold Music/Audio" else "🔇"
                st.markdown(
                    f"{icon} {hold['start']} - {hold['end']} "
                    f"({hold['duration']:.1f}s): **{hold['type']}** (RMS={hold['rms']:.4f})"
                )
        else:
            st.info("No extended hold segments detected or insufficient data.")


    st.markdown("---")
    st.subheader("📥 Download Reports")

    
    excel_buffer = DetailedReportGenerator.generate_excel_report(
        report,
        segments=st.session_state.get("transcript_segments") or [],
    )
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer,
        file_name=f"evaluation_report_{report.transcript_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tableau_buffer = generate_tableau_dataset_excel(report)
    st.download_button(
        label="📈 Download Tableau Dataset (Excel)",
        data=tableau_buffer,
        file_name=f"tableau_dataset_{report.transcript_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Includes all datapoints used to populate the UI (summary, criteria, issues, coaching, segments, holds, costs).",
    )


# ============================================================================
# S3 File Browser Section
# ============================================================================
st.markdown("---")
st.markdown("## ☁️ S3 Audio File Browser")
st.markdown("Browse and process audio files directly from your S3 bucket.")

# Initialize session state for S3 browser
if 's3_files' not in st.session_state:
    st.session_state['s3_files'] = []
if 'selected_files' not in st.session_state:
    st.session_state['selected_files'] = []
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = []

# S3 path configuration
col1, col2 = st.columns([3, 1])
with col1:
    s3_prefix = st.text_input(
        "S3 Folder Path",
        value="voice_analysis/voice_transcript/",
        help="Enter the folder path in your S3 bucket where audio files are stored"
    )
with col2:
    refresh_clicked = st.button("🔄 Refresh Files", type="primary")

# Refresh file list
if refresh_clicked:
    active_access_key = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
    active_secret_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
    active_region = aws_region or DEFAULT_AWS_REGION
    active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
    
    if not active_bucket:
        st.warning("⚠️ Please provide S3 Bucket Name in the sidebar.")
    elif not active_access_key or not active_secret_key:
        st.warning("⚠️ Please provide AWS Access Key ID and Secret Access Key in the sidebar.")
    else:
        with st.spinner("📂 Loading files from S3..."):
            try:
                session = create_aws_session(
                    region_name=active_region,
                    aws_access_key_id=active_access_key,
                    aws_secret_access_key=active_secret_key,
                )
                s3_client = session.client('s3', verify=False)
                files = list_s3_audio_files(active_bucket, s3_prefix, s3_client)
                st.session_state['s3_files'] = files
                st.session_state['selected_files'] = []
                if files:
                    st.success(f"✅ Found {len(files)} audio files")
                else:
                    st.info("No audio files found in this folder.")
            except Exception as e:
                st.error(f"Error accessing S3: {e}")

# Display file list if available
if st.session_state['s3_files']:
    files = st.session_state['s3_files']
    
    st.markdown(f"### 📁 Audio Files ({len(files)} files)")
    
    # Select all toggle
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Select All"):
            st.session_state['selected_files'] = [f['key'] for f in files]
            st.rerun()
    with col2:
        if st.button("❌ Deselect All"):
            st.session_state['selected_files'] = []
            st.rerun()
    with col3:
        st.markdown(f"**Selected:** {len(st.session_state['selected_files'])} / {len(files)}")
    
    # File list with checkboxes
    st.markdown("---")
    for i, file_info in enumerate(files):
        col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1.5])
        with col1:
            is_selected = file_info['key'] in st.session_state['selected_files']
            checkbox_label = f"Select {file_info.get('filename', 'file')}"
            if st.checkbox(
                checkbox_label,
                value=is_selected,
                key=f"file_checkbox_{i}",
                label_visibility="collapsed",
            ):
                if file_info['key'] not in st.session_state['selected_files']:
                    st.session_state['selected_files'].append(file_info['key'])
            else:
                if file_info['key'] in st.session_state['selected_files']:
                    st.session_state['selected_files'].remove(file_info['key'])
        with col2:
            st.markdown(f"**{file_info['filename']}**")
        with col3:
            st.markdown(f"{file_info['size_mb']} MB")
        with col4:
            st.markdown(f"📅 {file_info['last_modified'][:10]}")
    
    st.markdown("---")
    
    # Processing buttons
    selected_count = len(st.session_state['selected_files'])
    
    col1, col2 = st.columns(2)
    with col1:
        process_single = st.button(
            f"▶️ Process Selected ({selected_count} file{'s' if selected_count != 1 else ''})",
            type="primary",
            disabled=selected_count == 0
        )
    with col2:
        process_batch = st.button(
            f"🚀 Batch Process All ({len(files)} files)",
            type="secondary",
            disabled=len(files) == 0
        )
    
    # Process selected files
    if process_single and selected_count > 0:
        active_access_key = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
        active_secret_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
        active_region = aws_region or DEFAULT_AWS_REGION
        active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
        
        session = create_aws_session(
            region_name=active_region,
            aws_access_key_id=active_access_key,
            aws_secret_access_key=active_secret_key,
        )
        s3_client = session.client('s3', verify=False)
        transcribe_client = session.client('transcribe', verify=False)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file_key in enumerate(st.session_state['selected_files']):
            filename = file_key.split('/')[-1]
            status_text.markdown(f"⏳ Processing **{filename}** ({idx + 1}/{selected_count})...")
            
            result = process_s3_audio_file(
                bucket=active_bucket,
                file_key=file_key,
                s3_client=s3_client,
                transcribe_client=transcribe_client,
                aws_access_key_id=active_access_key,
                aws_secret_access_key=active_secret_key,
                aws_region=active_region,
                max_speakers=max_speakers,
                use_channel_id=use_channel_id,
                language_code=language_code,
                auto_detect_language=auto_detect_language,
                identify_multiple_languages=identify_multiple_languages,
                custom_vocabulary=custom_vocabulary if custom_vocabulary else None,
                prop_decrease=prop_decrease,
                run_parallel=run_parallel,
                translate_to_english=translate_to_english
            )
            results.append(result)
            progress_bar.progress((idx + 1) / selected_count)
        
        status_text.markdown("✅ **Processing Complete!**")
        st.session_state['batch_results'] = results
        # Generate + upload batch Excel once per run
        try:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state['batch_report_id'] = batch_id
            batch_excel_bytes = generate_batch_results_excel(results).getvalue()
            st.session_state['batch_excel_bytes'] = batch_excel_bytes

            if active_bucket:
                batch_key = f"{DEFAULT_EXCEL_REPORT_S3_PREFIX}batch_evaluation_results_{batch_id}.xlsx"
                _maybe_upload_excel_bytes_to_s3(
                    excel_bytes=batch_excel_bytes,
                    bucket=active_bucket,
                    object_key=batch_key,
                    s3_client=s3_client,
                    aws_access_key_id=active_access_key,
                aws_secret_access_key=active_secret_key,
                    aws_region=active_region,
                )
                st.session_state['batch_excel_s3_key'] = batch_key
        except Exception as e:
            logger.error(f"Error generating/uploading batch Excel: {e}")
    
    # Process all files (batch)
    if process_batch:
        active_access_key = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
        active_secret_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
        active_region = aws_region or DEFAULT_AWS_REGION
        active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
        
        session = create_aws_session(
            region_name=active_region,
            aws_access_key_id=active_access_key,
            aws_secret_access_key=active_secret_key,
        )
        s3_client = session.client('s3', verify=False)
        transcribe_client = session.client('transcribe', verify=False)
        
        all_keys = [f['key'] for f in files]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file_key in enumerate(all_keys):
            filename = file_key.split('/')[-1]
            status_text.markdown(f"⏳ Processing **{filename}** ({idx + 1}/{len(all_keys)})...")
            
            result = process_s3_audio_file(
                bucket=active_bucket,
                file_key=file_key,
                s3_client=s3_client,
                transcribe_client=transcribe_client,
                aws_access_key_id=active_access_key,
                aws_secret_access_key=active_secret_key,
                aws_region=active_region,
                max_speakers=max_speakers,
                use_channel_id=use_channel_id,
                language_code=language_code,
                auto_detect_language=auto_detect_language,
                identify_multiple_languages=identify_multiple_languages,
                custom_vocabulary=custom_vocabulary if custom_vocabulary else None,
                prop_decrease=prop_decrease,
                run_parallel=run_parallel,
                translate_to_english=translate_to_english
            )
            results.append(result)
            progress_bar.progress((idx + 1) / len(all_keys))
        
        status_text.markdown("✅ **Batch Processing Complete!**")
        st.session_state['batch_results'] = results
        # Generate + upload batch Excel once per run
        try:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state['batch_report_id'] = batch_id
            batch_excel_bytes = generate_batch_results_excel(results).getvalue()
            st.session_state['batch_excel_bytes'] = batch_excel_bytes

            if active_bucket:
                batch_key = f"{DEFAULT_EXCEL_REPORT_S3_PREFIX}batch_evaluation_results_{batch_id}.xlsx"
                _maybe_upload_excel_bytes_to_s3(
                    excel_bytes=batch_excel_bytes,
                    bucket=active_bucket,
                    object_key=batch_key,
                    s3_client=s3_client,
                    aws_access_key_id=active_access_key,
                aws_secret_access_key=active_secret_key,
                    aws_region=active_region,
                )
                st.session_state['batch_excel_s3_key'] = batch_key
        except Exception as e:
            logger.error(f"Error generating/uploading batch Excel: {e}")

# Display batch results
if st.session_state['batch_results']:
    st.markdown("---")
    st.markdown("### 📊 Batch Processing Results")
    
    results = st.session_state['batch_results']
    
    # Calculate totals for summary
    successful = [r for r in results if r['score'] is not None]
    total_cost = sum(
        r.get('cost_estimation', {}).get('total_cost', 0) 
        for r in results if r.get('cost_estimation')
    )
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = sum(r['score'] for r in successful) / len(successful) if successful else 0
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col2:
        st.metric("Success Rate", f"{len(successful)}/{len(results)}")
    with col3:
        st.metric("Total Est. Cost", f"${total_cost:.4f}")
    
    st.markdown("---")

    # Score graph per file/user
    st.markdown("#### 📈 Scores by File")
    if successful:
        labels = [r.get("filename", f"File {i+1}") for i, r in enumerate(successful)]
        scores = [float(r.get("score") or 0.0) for r in successful]
        display_labels = [
            (name[:22] + "…") if isinstance(name, str) and len(name) > 23 else name
            for name in labels
        ]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(scores)), scores, color="#1D4ED8")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        ax.set_xlabel("File / User")
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        try:
            st.pyplot(fig, width="stretch")
        except TypeError:
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("No successful evaluations to plot yet.")
    
    # Interactive results table with View buttons
    st.markdown("#### 📋 Results Summary")
    
    # Table header
    header_cols = st.columns([2.5, 1, 1.2, 1.5, 1.2, 0.8])
    header_cols[0].markdown("**File Name**")
    header_cols[1].markdown("**Score (%)**")
    header_cols[2].markdown("**Rating**")
    header_cols[3].markdown("**Status**")
    header_cols[4].markdown("**Cost Est.**")
    header_cols[5].markdown("**Action**")
    
    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    
    # Table rows
    for idx, r in enumerate(results):
        row_cols = st.columns([2.5, 1, 1.2, 1.5, 1.2, 0.8])
        
        # File name (truncate if too long)
        filename = r['filename']
        display_name = filename[:30] + "..." if len(filename) > 30 else filename
        row_cols[0].markdown(f"📄 {display_name}")
        
        # Score
        score_str = f"{r['score']:.1f}" if r['score'] else "N/A"
        row_cols[1].write(score_str)
        
        # Rating with color
        rating = r['rating'] if r['rating'] else "N/A"
        rating_colors = {
            "Desirable": "🟢",
            "Expected": "🔵", 
            "Basic": "🟡",
            "Undesirable": "🔴"
        }
        rating_icon = rating_colors.get(rating, "⚪")
        row_cols[2].write(f"{rating_icon} {rating}")
        
        # Status
        if r['evaluation_report']:
            row_cols[3].write("✅ Success")
        else:
            row_cols[3].write(f"❌ Error")
        
        # Cost estimation
        cost = r.get('cost_estimation', {})
        cost_str = cost.get('total_formatted', 'N/A') if cost else 'N/A'
        row_cols[4].write(cost_str)
        
        # View button
        if r['evaluation_report']:
            if row_cols[5].button("👁️", key=f"view_btn_{idx}", help=f"View details for {filename}"):
                st.session_state['selected_batch_file'] = r['filename']
                st.rerun()
        else:
            row_cols[5].write("-")
    
    st.markdown("---")
    
    # Download Excel report
    batch_id = st.session_state.get('batch_report_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_bytes = st.session_state.get('batch_excel_bytes')
    if not excel_bytes:
        excel_bytes = generate_batch_results_excel(results).getvalue()
        st.session_state['batch_excel_bytes'] = excel_bytes
    st.download_button(
        label="📥 Download Batch Results (Excel)",
        data=excel_bytes,
        file_name=f"batch_evaluation_results_{batch_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # View individual results with detailed evaluation
    st.markdown("### 📋 View Individual Results")
    
    # Create a selectbox to choose which file to view in detail
    file_options = ["Select a file to view details..."] + [r['filename'] for r in results if r['evaluation_report']]
    
    # Check if a file was selected via View button
    default_idx = 0
    if 'selected_batch_file' in st.session_state:
        try:
            default_idx = file_options.index(st.session_state['selected_batch_file'])
            # Clear the selection after using it
            del st.session_state['selected_batch_file']
        except ValueError:
            default_idx = 0
    
    selected_file = st.selectbox(
        "Choose file to view detailed evaluation:", 
        file_options, 
        index=default_idx,
        key="s3_file_selector"
    )
    
    if selected_file != "Select a file to view details...":
        # Find the selected result
        selected_result = next((r for r in results if r['filename'] == selected_file), None)
        
        if selected_result and selected_result['evaluation_report']:
            report = selected_result['evaluation_report']
            
            st.markdown(f"## 📄 Detailed Evaluation: {selected_file}")
            st.markdown("---")
            
            # Store audio data, segments, and metrics in session state for display functions
            st.session_state['current_s3_report'] = report
            st.session_state['current_s3_transcript'] = selected_result.get('transcript', '')
            st.session_state['transcript'] = selected_result.get('transcript', '')
            
            # Store audio data for voice clip playback
            if selected_result.get('audio_data') is not None:
                st.session_state['reduced_noise_y'] = selected_result['audio_data']
                st.session_state['sr'] = selected_result['sr']
                st.session_state['audio_cleaned'] = True
            
            # Store segments for evidence timestamp lookup
            st.session_state['transcript_segments'] = selected_result.get('segments', [])
            st.session_state['transcript_timing'] = selected_result.get('timing', {})
            st.session_state['audio_duration_seconds'] = selected_result.get('audio_duration_seconds', 0)
            
            # Store hold analysis and WPM
            st.session_state['hold_analysis'] = selected_result.get('hold_analysis', [])
            st.session_state['hold_summary'] = selected_result.get('hold_summary', {})
            st.session_state['cso_wpm'] = selected_result.get('cso_wpm')
            
            # ========== METRICS DISPLAY SECTION ==========
            st.markdown("### 📊 Call Metrics")
            
            hold_summary = selected_result.get('hold_summary', {}) or {}
            hold_analysis = selected_result.get('hold_analysis', []) or []
            cso_wpm = selected_result.get('cso_wpm')
            audio_duration = selected_result.get('audio_duration_seconds', 0)
            
            # Calculate hold tune usage status
            hold_music_time = hold_summary.get('hold_music', 0.0)
            dead_silence_time = hold_summary.get('dead_silence', 0.0)
            total_hold_time = hold_music_time + dead_silence_time
            
            # Determine if agent properly used hold tune
            if total_hold_time > 0:
                hold_tune_ratio = hold_music_time / total_hold_time
                if hold_tune_ratio >= 0.8:
                    hold_tune_status = "✅ Good - Hold tune used appropriately"
                    hold_tune_color = "#28a745"
                elif hold_tune_ratio >= 0.5:
                    hold_tune_status = "⚠️ Partial - Some dead silence detected"
                    hold_tune_color = "#2563EB"
                else:
                    hold_tune_status = "❌ Poor - Excessive dead silence instead of hold tune"
                    hold_tune_color = "#dc3545"
            else:
                hold_tune_status = "ℹ️ No extended hold periods detected"
                hold_tune_color = "#6c757d"
            
            # Display metrics in columns
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.9rem; color: #a0c4e8;">🎵 Hold Music Time</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{hold_music_time:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #5a1e1e 0%, #872d2d 100%); padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.9rem; color: #e8a0a0;">🔇 Dead Silence Time</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{dead_silence_time:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                wpm_display = f"{cso_wpm}" if cso_wpm is not None else "N/A"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e5a3a 0%, #2d8755 100%); padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.9rem; color: #a0e8c4;">🗣️ Words Per Minute</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{wpm_display} WPM</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m4:
                duration_display = f"{audio_duration/60:.1f} min" if audio_duration > 0 else "N/A"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4a1e5a 0%, #6d2d87 100%); padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 0.9rem; color: #d4a0e8;">⏱️ Call Duration</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{duration_display}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Hold Tune Usage Status
            st.markdown(f"""
            <div style="background: #1e1e2e; padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid {hold_tune_color};">
                <div style="font-size: 1rem; font-weight: bold; color: {hold_tune_color};">Hold Tune Usage Assessment</div>
                <div style="font-size: 0.95rem; color: #ccc; margin-top: 0.5rem;">{hold_tune_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed hold segments if any
            if hold_analysis:
                with st.expander("📋 View Hold/Silence Segments Detail"):
                    for hold in hold_analysis:
                        icon = "🎵" if hold["type"] == "Hold Music/Audio" else "🔇"
                        color = "#28a745" if hold["type"] == "Hold Music/Audio" else "#dc3545"
                        st.markdown(
                            f"<span style='color: {color}'>{icon} {hold['start']} - {hold['end']} "
                            f"({hold['duration']:.1f}s): **{hold['type']}** (RMS={hold['rms']:.4f})</span>",
                            unsafe_allow_html=True
                        )
            
            st.markdown("---")
            
            # Display the full evaluation report with override options
            display_evaluation_report(report)
            
            # Show transcript in expandable section
            with st.expander(f"📝 View Full Transcript - {selected_file}"):
                st.text_area(
                    "Transcript content",
                    selected_result.get('transcript', 'No transcript available'),
                    height=300,
                    key=f"detailed_transcript_{selected_result['key']}",
                    label_visibility="collapsed"
                )

st.markdown("---")

# ============================================================================
# Local File Upload Section
# ============================================================================
st.markdown("## 📤 Or Upload Local Audio File")
uploaded_file = st.file_uploader("🎵 Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:

  
    tab1, tab2, tab3 = st.tabs(["🎵 Audio Processing", "📝 Transcription", "✨ Quality Evaluation"])
    
    with tab1:
        st.subheader("🎵 Original Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Loading audio..."):
            y, sr = process_audio(uploaded_file)

        if y is not None:
            with st.expander("📊 Show Original Waveform"):
                st.pyplot(plot_waveform(y, sr, "Original Waveform"))
            
            if st.button("🧹 Clean Audio", type="primary"):
                with st.spinner("✨ Reducing noise... this might take a moment for long files."):
                    reduced_noise_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=False)
                    
                    st.session_state['reduced_noise_y'] = reduced_noise_y
                    st.session_state['sr'] = sr
                    st.session_state['audio_cleaned'] = True
                    
            if st.session_state.get('audio_cleaned'):
                reduced_noise_y = st.session_state['reduced_noise_y']
                sr = st.session_state['sr']
                
                buffer = io.BytesIO()
                sf.write(buffer, reduced_noise_y, sr, format='WAV', subtype='PCM_16')
                buffer.seek(0)
               
                st.session_state['cleaned_audio_bytes'] = buffer.getvalue()
                
                st.success("✅ Audio processed successfully!")
                
                st.subheader("🎵 Cleaned Audio")
                st.audio(buffer, format='audio/wav')
                
                with st.expander("📊 Show Cleaned Waveform"):
                    st.pyplot(plot_waveform(reduced_noise_y, sr, "Cleaned Waveform"))
                
                st.download_button(
                    label="📥 Download Cleaned Audio",
                    data=buffer,
                    file_name="cleaned_voice_note.wav",
                    mime="audio/wav"
                )
    
    with tab2:
        st.subheader("📝 Transcription")
        
       
        manual_transcript = st.text_area(
            "Or paste transcript manually:",
            height=200,
            placeholder="Paste your transcript here if you already have one..."
        )
        
        if manual_transcript:
            st.session_state['transcript'] = manual_transcript
            st.session_state['transcript_segments'] = []
            st.session_state['hold_analysis'] = []
            st.session_state['hold_summary'] = {"hold_music": 0.0, "dead_silence": 0.0}
            timing_info = compute_timing_from_segments(
                [],
                st.session_state.get('audio_duration_seconds')
            )
            st.session_state['transcript_timing'] = timing_info
            st.session_state['cso_wpm'] = compute_words_per_minute([], timing_info)
            st.success("✅ Transcript loaded!")
        
        if st.button("🎙️ Transcribe Audio", type="primary"):
         
            active_access_key = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
            active_secret_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
            active_region = aws_region or DEFAULT_AWS_REGION
            active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET

            if not active_bucket:
                st.warning("⚠️ Please provide S3 Bucket Name in the sidebar or .env.")
            elif not st.session_state.get('audio_cleaned'):
                st.warning("⚠️ Please clean the audio first in the Audio Processing tab.")
            else:
                try:
                    session = create_aws_session(
                        region_name=active_region,
                        aws_access_key_id=active_access_key,
                        aws_secret_access_key=active_secret_key,
                    )
                    s3_client = session.client('s3', verify=False)
                    transcribe_client = session.client('transcribe', verify=False)
                    
                  
                    reduced_noise_y = st.session_state['reduced_noise_y']
                    sr = st.session_state['sr']
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, reduced_noise_y, sr, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    
                    file_name = f"voice_note_{uuid.uuid4()}.wav"
                    
                    with st.spinner("☁️ Uploading to S3..."):
                        if upload_to_s3(buffer, active_bucket, file_name, s3_client):
                            st.success("✅ Uploaded to S3.")
                            
                            job_name = f"transcription_{uuid.uuid4()}"
                            s3_uri = f"s3://{active_bucket}/{file_name}"
                            
                            result = transcribe_audio(
                                s3_uri, 
                                job_name, 
                                transcribe_client, 
                                max_speakers=max_speakers,
                                use_channel_id=use_channel_id,
                                language_code=language_code,
                                auto_detect_language=auto_detect_language,
                                identify_multiple_languages=identify_multiple_languages,
                                custom_vocabulary=custom_vocabulary if custom_vocabulary else None
                            )
                            
                            if result:
                                transcript_text, segments, timing, language_info = result
                                st.session_state['transcript'] = transcript_text
                                st.session_state['transcript_segments'] = segments or []
                                st.session_state['detected_languages'] = (language_info or {}).get('detected_languages', [])
                                
                                timing_info = timing or compute_timing_from_segments(
                                    segments or [],
                                    st.session_state.get('audio_duration_seconds')
                                )
                                st.session_state['transcript_timing'] = timing_info
                                
                                st.session_state['cso_wpm'] = compute_words_per_minute(
                                    segments or [],
                                    timing_info
                                )

                                holds = detect_hold_segments(
                                    segments or [],
                                    st.session_state.get('reduced_noise_y'),
                                    st.session_state.get('sr'),
                                    min_gap=10.0
                                )
                                st.session_state['hold_analysis'] = holds
                                st.session_state['hold_summary'] = summarize_hold_segments(holds)

                                st.success("✅ Transcription Complete!")
                                
                                # Cleanup S3 object
                                try:
                                    s3_client.delete_object(Bucket=active_bucket, Key=file_name)
                                except:
                                    pass
                except Exception as e:
                    logger.error(f"An error occurred during transcription process: {e}")
                    st.error(f"An error occurred: {e}")
        
    
        if st.session_state.get('transcript'):
            st.markdown("### 📄 Transcript")
           
            st.text_area(
                "Call transcript", 
                st.session_state['transcript'],
                height=300,
                key="transcript_display",
                label_visibility="collapsed",
            )

            st.markdown("**Call Timing & Speech Metrics**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            timing_info = st.session_state.get('transcript_timing', {}) or {}
            hold_summary = st.session_state.get('hold_summary', {}) or {}
            cso_wpm = st.session_state.get('cso_wpm')

            start_ts = format_timestamp(timing_info.get('start_time', 0.0)) if timing_info else "--:--"
            end_ts = format_timestamp(timing_info.get('end_time', timing_info.get('duration', 0.0))) if timing_info else "--:--"
            duration_ts = format_timestamp(timing_info.get('duration', st.session_state.get('audio_duration_seconds', 0.0))) if timing_info else "--:--"

            with stats_col1:
                st.write(f"🕒 Start: {start_ts} | End: {end_ts}")
                st.write(f"⏱️ Duration: {duration_ts}")

            with stats_col2:
                st.write(
                    f"🎵 Hold Music: {hold_summary.get('hold_music', 0.0):.1f}s\n\n"
                    f"🔇 Dead Silence: {hold_summary.get('dead_silence', 0.0):.1f}s"
                )

            with stats_col3:
                wpm_display = f"{cso_wpm} WPM" if cso_wpm is not None else "N/A"
                st.write(f"🗣️ CSO Speech Rate: {wpm_display}")

            segments_timeline = st.session_state.get('transcript_segments') or []
            st.markdown("**Conversation Timeline (per speaker turn)**")
            if segments_timeline:
                for seg in segments_timeline:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start))
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "").strip()
                    st.markdown(f"- {format_time_range(start, end)} — **{speaker}**: {text}")
            else:
                st.info("No timed segments available for this transcript.")
    
    with tab3:
        st.subheader("✨ AI Quality Evaluation")
        st.markdown("""
        This evaluation uses **multiple specialized AI agents** to assess the call quality:
        
        - 🎯 **Opening Greeting Agent** - Evaluates call opening
        - 🔐 **Verification Agent** - Checks identity verification
        - 💬 **Soft Skills Agent** - Assesses communication skills
        - 📋 **Enquiry Resolution Agent** - Evaluates problem-solving
        - 📈 **Cross-Selling Agent** - Checks sales opportunities
        - 👋 **Wrap-Up Agent** - Evaluates call closing
        
        All agents use **AWS Bedrock Claude Sonnet 3.5** for evaluation.
        """)
        
        if st.button("🚀 Run Quality Evaluation", type="primary"):
            if not st.session_state.get('transcript'):
                st.warning("⚠️ Please transcribe the audio first or paste a transcript manually.")
            else:
                active_access_key = aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID
                active_secret_key = aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY
                active_region = aws_region or DEFAULT_AWS_REGION

                if not active_access_key or not active_secret_key:
                    st.warning("⚠️ Please provide AWS Access Key ID and Secret Access Key in the sidebar.")
                else:
                    report = run_quality_evaluation(
                        st.session_state['transcript'],
                        active_access_key,
                        active_secret_key,
                        active_region,
                        translate_to_english=translate_to_english,
                    )
                    
                    if report:
                        st.session_state['evaluation_report'] = report
                        # Upload single-file Excel report to S3 (voice_analysis/excel report/)
                        try:
                            active_bucket = st.session_state.get("s3_bucket") or DEFAULT_S3_BUCKET
                            if active_bucket:
                                from orchestrator import DetailedReportGenerator

                                excel_bytes = DetailedReportGenerator.generate_excel_report(
                                    report,
                                    segments=st.session_state.get("transcript_segments") or [],
                                ).getvalue()
                                ts = _timestamp_for_filename(getattr(report, "evaluation_timestamp", ""))
                                base_name = _safe_s3_key_part(getattr(report, "transcript_id", "evaluation"))
                                object_key = f"{DEFAULT_EXCEL_REPORT_S3_PREFIX}evaluation_report_{base_name}_{ts}.xlsx"
                                uploaded = _maybe_upload_excel_bytes_to_s3(
                                    excel_bytes=excel_bytes,
                                    bucket=active_bucket,
                                    object_key=object_key,
                                    aws_access_key_id=active_access_key,
                                    aws_secret_access_key=active_secret_key,
                                    aws_region=active_region,
                                )
                                if uploaded:
                                    st.session_state["single_excel_s3_key"] = object_key
                        except Exception as e:
                            logger.error(f"Error generating/uploading single-file Excel report: {e}")
                        display_evaluation_report(report)
        
        elif st.session_state.get('evaluation_report'):
            display_evaluation_report(st.session_state['evaluation_report'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    <p>✨ AI-Powered Voice Quality Evaluator | Using AWS Bedrock Claude Sonnet 3.5</p>
    <p>Built with Streamlit, LangChain, and AWS Services</p>
</div>
""", unsafe_allow_html=True)
