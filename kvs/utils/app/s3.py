"""
S3 utility functions
"""

import yaml
from io import BytesIO
from os import getenv

import boto3
import pandas as pd

AWS_CREDS = None
try:
    AWS_CREDS = yaml.load(open("kinesis.yml", "r"), Loader=yaml.SafeLoader)
except FileNotFoundError:
    pass


def s3_read(bucket, key):
    """Read file from S3 as bytes."""
    client = (
        boto3.client(
            "s3",
            region_name="ap-southeast-1",
            aws_access_key_id=AWS_CREDS["KEY_ID"],
            aws_secret_access_key=AWS_CREDS["ACCESS_KEY"],
        )
        if AWS_CREDS is not None
        else boto3.client("s3")
    )
    obj = client.get_object(Bucket=bucket, Key=key)
    return BytesIO(obj["Body"].read())


def s3_read_parquet(bucket, key):
    """Read parquet from S3."""
    try:
        return pd.read_parquet(s3_read(bucket, key))
    except Exception:
        return
