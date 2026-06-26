import os

import boto3
from dotenv import load_dotenv

load_dotenv()

DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
DEFAULT_AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
DEFAULT_AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()


def create_aws_session(
    *,
    region_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> boto3.Session:
    """Create a boto3 session using IAM access keys."""
    region = (region_name or DEFAULT_AWS_REGION or "ap-southeast-1").strip()
    access_key = (aws_access_key_id or DEFAULT_AWS_ACCESS_KEY_ID or "").strip()
    secret_key = (aws_secret_access_key or DEFAULT_AWS_SECRET_ACCESS_KEY or "").strip()

    if access_key and secret_key:
        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    return boto3.Session(region_name=region)
