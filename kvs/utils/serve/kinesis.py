from datetime import datetime
from os import getenv
from typing import Optional
from urllib.parse import unquote

import boto3
import yaml

region_name = "ap-southeast-1"
try:
    kinesis = yaml.load(open("kinesis.yml", "r"), Loader=yaml.SafeLoader)
    aws_secret_access_key = kinesis.get("ACCESS_KEY")
    aws_access_key_id = kinesis.get("KEY_ID")
except FileNotFoundError:
    aws_secret_access_key = getenv("ACCESS_KEY") or None
    if aws_secret_access_key:
        aws_secret_access_key = unquote(aws_secret_access_key)
    aws_access_key_id = getenv("KEY_ID") or None
    if aws_access_key_id:
        aws_access_key_id = unquote(aws_access_key_id)

kvs = boto3.client(
    "kinesisvideo",
    region_name=region_name,
    aws_secret_access_key=aws_secret_access_key,
    aws_access_key_id=aws_access_key_id,
)


class CameraStream:
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        # Grab the HTTP Live Streaming endpoint
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kinesisvideo.html#KinesisVideo.Client.get_data_endpoint
        self.endpoint = kvs.get_data_endpoint(
            StreamName=self.stream_name,
            APIName="GET_HLS_STREAMING_SESSION_URL",
        )["DataEndpoint"]
        self.clip_endpoint = kvs.get_data_endpoint(
            StreamName=self.stream_name,
            APIName="GET_CLIP",
        )["DataEndpoint"]

    def get_clip(self, start: datetime, end: datetime):
        kvam = boto3.client(
            "kinesis-video-archived-media",
            endpoint_url=self.clip_endpoint,
            region_name=region_name,
            aws_secret_access_key=aws_secret_access_key,
            aws_access_key_id=aws_access_key_id,
        )
        payload = kvam.get_clip(
            StreamName=self.stream_name,
            ClipFragmentSelector={
                "FragmentSelectorType": "SERVER_TIMESTAMP",
                "TimestampRange": {
                    "StartTimestamp": start,
                    "EndTimestamp": end,
                },
            },
        )["Payload"]
        return payload

    def get_stream_url(self, start: Optional[datetime] = None):
        kvam = boto3.client(
            "kinesis-video-archived-media",
            endpoint_url=self.endpoint,
            region_name=region_name,
            aws_secret_access_key=aws_secret_access_key,
            aws_access_key_id=aws_access_key_id,
        )
        # Create a local streaming session
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kinesis-video-archived-media.html#KinesisVideoArchivedMedia.Client.get_hls_streaming_session_url
        if start:
            url = kvam.get_hls_streaming_session_url(
                StreamName=self.stream_name,
                PlaybackMode="LIVE_REPLAY",
                HLSFragmentSelector={
                    "FragmentSelectorType": "SERVER_TIMESTAMP",
                    "TimestampRange": {
                        "StartTimestamp": start,
                    },
                },
                DisplayFragmentTimestamp="ALWAYS",
                # 12 hours is the maximum session duration
                Expires=43200,
            )["HLSStreamingSessionURL"]
        else:
            url = kvam.get_hls_streaming_session_url(
                StreamName=self.stream_name,
                PlaybackMode="LIVE",
            )["HLSStreamingSessionURL"]

        return url


CAMERA_NAMES = []
DEVICES = [CameraStream(c) for c in CAMERA_NAMES]
