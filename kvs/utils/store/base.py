from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class PredictionStore(ABC):
    @abstractmethod
    def _list_buckets(self, prefix: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def _read_bucket(self, path: str) -> str:
        raise NotImplementedError

    def _read_json(self, prefix: str) -> pd.DataFrame:
        buckets = self._list_buckets(prefix)
        if len(buckets) == 0:
            return pd.DataFrame()
        # Pandas use gcsfs which isn't available on k8s, hence downloading locally.
        local = [self._read_bucket(path) for path in buckets]
        # Load all buckets into memory as json dataframe
        df = [pd.read_json(path_or_buf=fp, lines=True) for fp in local]
        # Merge logs from all matched buckets
        return pd.concat(df, ignore_index=True)

    def load_predictions(self, prefix: str) -> pd.DataFrame:
        """Fetches logged predictions from all buckets that match the given prefix.

        :param prefix: Prefix of the bucket object
        :type prefix: str
        :return: All items in matched buckets
        :rtype: pd.DataFrame
        """
        return self._read_json(prefix)

    def load_labels(self, prefix: str) -> pd.DataFrame:
        """Fetches ground truth labels from the given path.

        If the prediction id is a composite key of the format: {server_id}/{created_at}/{entity_id},
        overrides the corresponding columns to the result set.

        :return: All items in the label bucket
        :rtype: pd.DataFrame
        """
        labels = self._read_json(prefix)

        # Split up prediction id into bucket identifiers
        keys = list(zip(*map(lambda l: l.split("/"), labels["prediction_id"])))
        if len(keys) == 3:
            labels["server_id"] = keys[0]
            labels["created_at"] = pd.to_datetime(keys[1])
            labels["entity_id"] = keys[2]

        return labels
