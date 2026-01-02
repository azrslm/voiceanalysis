

import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import io
import boto3
import time
import uuid
import requests
import os
import logging
import json
import math
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


DEFAULT_AWS_PROFILE = os.getenv("AWS_PROFILE", "income-adfs")
DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
DEFAULT_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# AWS Pricing (ap-southeast-1 region)
TRANSCRIBE_PRICE_PER_MINUTE = 0.024  # USD per minute
BEDROCK_INPUT_PRICE_PER_1K = 0.003   # USD per 1K input tokens (Claude 3.5 Sonnet)
BEDROCK_OUTPUT_PRICE_PER_1K = 0.015  # USD per 1K output tokens (Claude 3.5 Sonnet)


st.set_page_config(
    page_title="✨ Voice Quality Evaluator",
    page_icon="🎙️",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Box Styles */
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        border-left: 5px solid #e94560;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #1a1a2e;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-value.score {
        color: #28a745;
        font-size: 2.2rem;
    }
    
    /* Card Styles */
    .report-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Findings & Evidence */
    .finding-box {
        margin-bottom: 1rem;
        color: #333;
    }
    .evidence-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-style: italic;
        color: #0d47a1;
    }
    
    /* Badge Styles */
    .rating-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-Expected { background-color: #d4edda; color: #155724; }
    .badge-Desirable { background-color: #c3e6cb; color: #155724; }
    .badge-Basic { background-color: #fff3cd; color: #856404; }
    .badge-Undesirable { background-color: #f8d7da; color: #721c24; }
    
    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Transcript */
    .transcript-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .transcript-line {
        padding: 0.25rem 0;
        border-bottom: 1px solid #eee;
    }
    .speaker-label {
        font-weight: bold;
        color: #0f3460;
    }
    
    /* Override Dropdown Customization */
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎙️ Voice Note Quality Evaluator</h1>
    <p>✨ AI-Powered noise reduction, transcription, and call quality evaluation</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.header("🔐 AWS Configuration")

with st.sidebar.expander("AWS Settings", expanded=False):
    aws_profile = st.text_input(
        "AWS Profile Name",
        key="aws_profile",
        value=DEFAULT_AWS_PROFILE
    )
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

with st.sidebar.expander("Transcription Settings", expanded=False):
    max_speakers = st.number_input("Max Speakers (for Diarization)", min_value=2, max_value=10, value=2)
    
    # Channel Identification - best for stereo audio with agent/customer on separate channels
    use_channel_id = st.checkbox(
        "Use Channel Identification", 
        value=False,
        help="Enable if audio is STEREO with agent on one channel and customer on another. This gives 100% accurate speaker identification."
    )
    
    # Language selection
    language_code = st.selectbox(
        "Language",
        options=["en-US", "en-GB", "en-AU", "en-IN", "en-SG"],
        index=0,
        help="Select the primary language of the audio"
    )
    
    # Custom vocabulary (optional)
    custom_vocabulary = st.text_input(
        "Custom Vocabulary Name (optional)",
        value="",
        help="Enter the name of a pre-created AWS Transcribe custom vocabulary"
    )
    
    # Show speaker labels info
    if use_channel_id:
        st.info("📢 Channel ID mode: Left channel = Agent, Right channel = Customer")
    else:
        st.info(f"📢 Speaker diarization mode: Up to {max_speakers} speakers will be detected")


st.sidebar.markdown("---")
st.sidebar.header("🎚️ Audio Settings")
prop_decrease = st.sidebar.slider("Noise Reduction Strength", 0.0, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("📊 Evaluation Settings")
run_parallel = st.sidebar.checkbox("Run Agents in Parallel", value=True)


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


def process_s3_audio_file(
    bucket, 
    file_key, 
    s3_client, 
    transcribe_client,
    aws_profile,
    aws_region,
    max_speakers=2,
    use_channel_id=False,
    language_code='en-US',
    custom_vocabulary=None,
    prop_decrease=0.75,
    run_parallel=True
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
        'error': None
    }
    
    try:
        # Download audio file
        audio_buffer = download_s3_file(bucket, file_key, s3_client)
        if audio_buffer is None:
            result['error'] = "Failed to download file"
            return result
        
        # Load and process audio
        y, sr = librosa.load(audio_buffer, sr=None)
        
        # Apply noise reduction
        reduced_noise_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=False)
        
        # Save to buffer for upload
        processed_buffer = io.BytesIO()
        sf.write(processed_buffer, reduced_noise_y, sr, format='WAV')
        processed_buffer.seek(0)
        
        # Upload processed file to S3 for transcription
        temp_key = f"temp_processed/{uuid.uuid4()}.wav"
        if not upload_to_s3(processed_buffer, bucket, temp_key, s3_client):
            result['error'] = "Failed to upload processed audio"
            return result
        
        # Transcribe
        job_name = f"batch_transcription_{uuid.uuid4()}"
        s3_uri = f"s3://{bucket}/{temp_key}"
        
        transcript_result = transcribe_audio(
            s3_uri, 
            job_name, 
            transcribe_client, 
            max_speakers=max_speakers,
            use_channel_id=use_channel_id,
            language_code=language_code,
            custom_vocabulary=custom_vocabulary
        )
        
        # Cleanup temp file
        try:
            s3_client.delete_object(Bucket=bucket, Key=temp_key)
        except:
            pass
        
        if transcript_result is None:
            result['error'] = "Transcription failed"
            return result
        
        transcript_text, segments, timing = transcript_result
        result['transcript'] = transcript_text
        
        # Run quality evaluation
        from orchestrator import create_orchestrator
        
        orchestrator = create_orchestrator(
            profile_name=aws_profile,
            region_name=aws_region,
            model_id=DEFAULT_BEDROCK_MODEL_ID
        )
        
        report = orchestrator.evaluate_transcript(
            transcript=transcript_text,
            transcript_id=filename,
            parallel=run_parallel
        )
        
        if report:
            result['evaluation_report'] = report
            result['score'] = round(report.percentage_score, 1)
            result['rating'] = report.overall_rating
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        result['error'] = str(e)
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
        summary_rows.append({
            'File Name': r['filename'],
            'S3 Key': r['key'],
            'Score (%)': r['score'] if r['score'] else 'N/A',
            'Rating': r['rating'] if r['rating'] else 'N/A',
            'Status': 'Success' if r['evaluation_report'] else f"Error: {r['error']}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Detailed results per file
    detailed_rows = []
    for r in results:
        if r['evaluation_report']:
            report = r['evaluation_report']
            for agent_name, agent_result in report.agent_results.items():
                for eval in agent_result.evaluations:
                    detailed_rows.append({
                        'File Name': r['filename'],
                        'Agent Category': agent_result.agent_name,
                        'Criteria ID': eval.criteria_id,
                        'Criteria Name': eval.criteria_name,
                        'Rating': eval.rating,
                        'Score': eval.score,
                        'Max Score': eval.max_score,
                        'Weight (%)': eval.weight
                    })
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Write to Excel with multiple sheets
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        if not detailed_df.empty:
            detailed_df.to_excel(writer, index=False, sheet_name='Detailed Results')
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
            
            # Map speaker labels to roles for evaluation
            # spk_0 is typically the first speaker (agent) in call center scenarios
            speaker_role = "CSO" if speaker == "spk_0" else "Customer"
            timestamp = format_timestamp(start_time)
            
            transcript_lines.append(f"{timestamp} **{speaker_role}**: {text}")
            segments_info.append({
                "speaker": speaker_role,
                "raw_speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": text
            })
            
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


def transcribe_audio(s3_uri, job_name, transcribe_client, max_speakers=2, 
                     use_channel_id=False, language_code='en-US', 
                     custom_vocabulary=None):
    """
    Transcribe audio using AWS Transcribe with enhanced settings.
    
    Args:
        s3_uri: S3 URI of the audio file
        job_name: Unique job name for transcription
        transcribe_client: boto3 transcribe client
        max_speakers: Maximum number of speakers for diarization (2-10)
        use_channel_id: If True, use channel identification (for stereo audio)
        language_code: Language code (e.g., 'en-US', 'en-GB', 'en-SG')
        custom_vocabulary: Name of custom vocabulary (optional)
    
    Returns:
        Tuple of (transcript_text, segments, timing_info) or None
    """
    logger.info(f"Starting Transcription job: JobName={job_name}, URI={s3_uri}, "
                f"MaxSpeakers={max_speakers}, ChannelID={use_channel_id}, Language={language_code}")
    
    try:
        # Build settings based on mode
        settings = {}
        
        if use_channel_id:
            # Channel Identification mode - for stereo audio with agent/customer on separate channels
            # This provides 100% accurate speaker identification
            settings['ChannelIdentification'] = True
            logger.info("Using Channel Identification mode (stereo audio)")
        else:
            # Speaker Diarization mode - for mono audio
            settings['ShowSpeakerLabels'] = True
            settings['MaxSpeakerLabels'] = max_speakers
            logger.info(f"Using Speaker Diarization mode with max {max_speakers} speakers")
        
        # Add custom vocabulary if provided
        if custom_vocabulary and custom_vocabulary.strip():
            settings['VocabularyName'] = custom_vocabulary.strip()
            logger.info(f"Using custom vocabulary: {custom_vocabulary}")
        
        # Start transcription job with enhanced settings
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='wav',
            LanguageCode=language_code,
            Settings=settings
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
            response = requests.get(transcript_uri)
            return parse_transcript(response.json(), use_channel_id=use_channel_id)
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
        y, sr = librosa.load(uploaded_file, sr=None)
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
    sf.write(buffer, snippet, sr, format='WAV')
    buffer.seek(0)
    return buffer


def run_quality_evaluation(transcript_text, aws_profile, aws_region):
    """Run the multi-agent quality evaluation on the transcript"""
    try:
        from orchestrator import create_orchestrator, DetailedReportGenerator
        
       
        aws_profile = aws_profile or DEFAULT_AWS_PROFILE
        aws_region = aws_region or DEFAULT_AWS_REGION
        bedrock_model_id = DEFAULT_BEDROCK_MODEL_ID

        
        orchestrator = create_orchestrator(
            profile_name=aws_profile,
            region_name=aws_region,
            model_id=bedrock_model_id
        )
        
      
        with st.spinner("✨ AI agents are evaluating the transcript..."):
            report = orchestrator.evaluate_transcript(
                transcript=transcript_text,
                parallel=run_parallel
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


def display_evaluation_report(report):
    """Display the evaluation report in the new UI style"""
    from orchestrator import DetailedReportGenerator
    
    timing_info = st.session_state.get('transcript_timing', {}) or {}
    hold_summary = st.session_state.get('hold_summary', {}) or {}
    cso_wpm = st.session_state.get('cso_wpm')

 
    st.markdown("###")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        
        if report.percentage_score >= 80:
            score_color = "#28a745" # Green
        elif report.percentage_score >= 60:
            score_color = "#fd7e14" # Orange
        else:
            score_color = "#dc3545" # Red

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Score</div>
            <div class="metric-value score" style="color: {score_color}">{report.total_score:.1f} / {report.max_possible_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        agent_name = getattr(report, "detected_agent_name", "") or "Unknown"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Agent</div>
            <div class="metric-value">{agent_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        compliance_status = "Passed" if report.percentage_score >= 85 else "Review Needed"
        color = "#28a745" if compliance_status == "Passed" else "#ffc107"
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
            <div class="metric-container" style="border-left-color: #fd7e14;">
                <div class="metric-label">Token Breakdown</div>
                <div class="metric-value" style="font-size: 1rem;">In: {cost_breakdown['bedrock']['input_tokens']:,}</div>
                <div class="metric-label" style="font-size: 0.75rem;">Out: {cost_breakdown['bedrock']['output_tokens']:,}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Cost estimation requires audio duration and transcript data.")

    st.markdown("---")

    

    col_left, col_right = st.columns([2, 1])
    
   
    with col_left:
        st.markdown('<div class="section-header">📝 Detailed Scorecard</div>', unsafe_allow_html=True)
        
        
        all_evaluations = []
        for agent_name, result in report.agent_results.items():
            all_evaluations.extend(result.evaluations)
            
      
        for i, eval in enumerate(all_evaluations):
            
            unique_key = f"{eval.criteria_id}_{i}"
            
            
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
                    st.markdown("**Rating:**")
                    st.markdown(f"<span class='rating-badge badge-{eval.rating}'>{eval.rating}</span>", unsafe_allow_html=True)
                    
                    st.markdown("<br>**Override AI:**", unsafe_allow_html=True)
                    
                    
                    current_rating = st.session_state.get(f"override_{unique_key}", eval.rating)
                    
                    new_rating = st.selectbox(
                        "Select Rating",
                        options=["Expected", "Desirable", "Basic", "Undesirable"],
                        index=["Expected", "Desirable", "Basic", "Undesirable"].index(current_rating) if current_rating in ["Expected", "Desirable", "Basic", "Undesirable"] else 0,
                        key=f"override_select_{unique_key}",
                        label_visibility="collapsed"
                    )
                    
                    \
    with col_right:
        st.markdown('<div class="section-header">🎧 Audio & Transcript</div>', unsafe_allow_html=True)
        
      
        if st.session_state.get('audio_cleaned'):
            try:
                y_clean = st.session_state.get('reduced_noise_y')
                sr = st.session_state.get('sr')
                if y_clean is not None:
                    buffer = io.BytesIO()
                    sf.write(buffer, y_clean, sr, format='WAV')
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

    
    excel_buffer = DetailedReportGenerator.generate_excel_report(report)
    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer,
        file_name=f"evaluation_report_{report.transcript_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
    active_profile = aws_profile or DEFAULT_AWS_PROFILE
    active_region = aws_region or DEFAULT_AWS_REGION
    active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
    
    if not active_bucket:
        st.warning("⚠️ Please provide S3 Bucket Name in the sidebar.")
    else:
        with st.spinner("📂 Loading files from S3..."):
            try:
                session = boto3.Session(
                    profile_name=active_profile,
                    region_name=active_region
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
            if st.checkbox("", value=is_selected, key=f"file_checkbox_{i}"):
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
        active_profile = aws_profile or DEFAULT_AWS_PROFILE
        active_region = aws_region or DEFAULT_AWS_REGION
        active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
        
        session = boto3.Session(
            profile_name=active_profile,
            region_name=active_region
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
                aws_profile=active_profile,
                aws_region=active_region,
                max_speakers=max_speakers,
                use_channel_id=use_channel_id,
                language_code=language_code,
                custom_vocabulary=custom_vocabulary if custom_vocabulary else None,
                prop_decrease=prop_decrease,
                run_parallel=run_parallel
            )
            results.append(result)
            progress_bar.progress((idx + 1) / selected_count)
        
        status_text.markdown("✅ **Processing Complete!**")
        st.session_state['batch_results'] = results
    
    # Process all files (batch)
    if process_batch:
        active_profile = aws_profile or DEFAULT_AWS_PROFILE
        active_region = aws_region or DEFAULT_AWS_REGION
        active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET
        
        session = boto3.Session(
            profile_name=active_profile,
            region_name=active_region
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
                aws_profile=active_profile,
                aws_region=active_region,
                max_speakers=max_speakers,
                use_channel_id=use_channel_id,
                language_code=language_code,
                custom_vocabulary=custom_vocabulary if custom_vocabulary else None,
                prop_decrease=prop_decrease,
                run_parallel=run_parallel
            )
            results.append(result)
            progress_bar.progress((idx + 1) / len(all_keys))
        
        status_text.markdown("✅ **Batch Processing Complete!**")
        st.session_state['batch_results'] = results

# Display batch results
if st.session_state['batch_results']:
    st.markdown("---")
    st.markdown("### 📊 Batch Processing Results")
    
    results = st.session_state['batch_results']
    
    # Summary table
    summary_data = []
    for r in results:
        status = "✅ Success" if r['evaluation_report'] else f"❌ {r['error']}"
        summary_data.append({
            "File": r['filename'],
            "Score (%)": r['score'] if r['score'] else "N/A",
            "Rating": r['rating'] if r['rating'] else "N/A",
            "Status": status
        })
    
    import pandas as pd
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)
    
    # Calculate averages
    successful = [r for r in results if r['score'] is not None]
    if successful:
        avg_score = sum(r['score'] for r in successful) / len(successful)
        st.markdown(f"**Average Score:** {avg_score:.1f}% | **Success Rate:** {len(successful)}/{len(results)}")
    
    # Download Excel report
    st.markdown("---")
    excel_buffer = generate_batch_results_excel(results)
    st.download_button(
        label="📥 Download Batch Results (Excel)",
        data=excel_buffer,
        file_name=f"batch_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # View individual results
    with st.expander("📋 View Individual Transcripts & Reports"):
        for r in results:
            if r['evaluation_report']:
                st.markdown(f"#### 📄 {r['filename']}")
                st.markdown(f"**Score:** {r['score']}% | **Rating:** {r['rating']}")
                with st.expander(f"View Transcript - {r['filename']}"):
                    st.text_area("", r['transcript'], height=200, key=f"transcript_{r['key']}")

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
                sf.write(buffer, reduced_noise_y, sr, format='WAV')
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
         
            active_profile = aws_profile or DEFAULT_AWS_PROFILE
            active_region = aws_region or DEFAULT_AWS_REGION
            active_bucket = s3_bucket_name or DEFAULT_S3_BUCKET

            if not active_bucket:
                st.warning("⚠️ Please provide S3 Bucket Name in the sidebar or .env.")
            elif not st.session_state.get('audio_cleaned'):
                st.warning("⚠️ Please clean the audio first in the Audio Processing tab.")
            else:
                try:
                    session = boto3.Session(
                        profile_name=active_profile,
                        region_name=active_region
                    )
                    s3_client = session.client('s3', verify=False)
                    transcribe_client = session.client('transcribe', verify=False)
                    
                  
                    reduced_noise_y = st.session_state['reduced_noise_y']
                    sr = st.session_state['sr']
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, reduced_noise_y, sr, format='WAV')
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
                                custom_vocabulary=custom_vocabulary if custom_vocabulary else None
                            )
                            
                            if result:
                                transcript_text, segments, timing = result
                                st.session_state['transcript'] = transcript_text
                                st.session_state['transcript_segments'] = segments or []
                                
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
                active_profile = aws_profile or DEFAULT_AWS_PROFILE
                active_region = aws_region or DEFAULT_AWS_REGION

                report = run_quality_evaluation(
                    st.session_state['transcript'],
                    active_profile,
                    active_region
                )
                    
                if report:
                    st.session_state['evaluation_report'] = report
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
