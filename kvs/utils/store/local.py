from glob import glob
from typing import List

from .base import PredictionStore


class LocalPredictionStore(PredictionStore):
    def _read_bucket(self, path: str) -> str:
        return path

    def _list_buckets(self, prefix: str) -> List[str]:
        return glob(f"{prefix}*")
