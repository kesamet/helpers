from dataclasses import dataclass
from typing import List, Optional

from boto3 import client

from .base import PredictionStore


@dataclass(frozen=True)
class AWSCredentials:
    region_name: str
    aws_secret_access_key: str
    aws_access_key_id: str


class S3PredictionStore(PredictionStore):
    def __init__(self, bucket: str, credentials: Optional[AWSCredentials] = None):
        self._bucket: str = bucket
        self._client: client = (
            client(
                "s3",
                region_name=credentials.region_name,
                aws_secret_access_key=credentials.aws_secret_access_key,
                aws_access_key_id=credentials.aws_access_key_id,
            )
            if credentials is not None
            else client("s3")
        )

    def _read_bucket(self, path: str) -> str:
        fp = f"./{path.split('/')[-1]}"
        self._client.download_file(Bucket=self._bucket, Key=path, Filename=fp)
        return fp

    def _list_buckets(self, prefix: str) -> List[str]:
        resp = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        files = resp.get("Contents", [])
        # Filter out directories from object list
        keys = [f["Key"] for f in files if not f["Key"].endswith("/")]
        return keys
