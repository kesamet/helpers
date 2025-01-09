from typing import List

from google.cloud.storage import Client

from .base import PredictionStore


class GCSPredictionStore(PredictionStore):
    def __init__(self, bucket: str):
        self._bucket: str = bucket
        self._client: Client = Client()

    def _read_bucket(self, path: str) -> str:
        fp = f"./{path.split('/')[-1]}"
        with open(fp, "wb") as f:
            self._client.download_blob_to_file(path, f)
        return fp

    def _list_buckets(self, prefix: str) -> List[str]:
        files = self._client.list_blobs(bucket_or_name=self._bucket, prefix=prefix)
        paths = [f"gs://{self._bucket}/{blob.name}" for blob in files]
        return paths
