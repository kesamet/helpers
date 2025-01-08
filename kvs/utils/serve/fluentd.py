import json
from datetime import datetime
from os import getenv

import pandas as pd
from fluent.sender import FluentSender
from spanlib.infrastructure.kubernetes.env_var import (
    BEDROCK_FLUENTD_ADDR,
    BEDROCK_FLUENTD_PREFIX,
)


class PandasExporter:
    def __init__(self, exec_date: datetime):
        # Environment variables will be injected by Bedrock
        pod_name = getenv("BEDROCK_PIPELINE_RUN_ID", "unknown-run")
        fluentd_prefix = getenv(BEDROCK_FLUENTD_PREFIX, "models.predictions")
        tag = f"{fluentd_prefix}.{exec_date.isoformat()}.{pod_name}"

        fluentd_server = getenv(
            BEDROCK_FLUENTD_ADDR, "fluentd-logging.core.svc.cluster.local"
        ).split(":")
        fluentd_port = int(fluentd_server[1]) if len(fluentd_server) > 1 else 24224

        self._sender: FluentSender = FluentSender(
            tag=tag,
            host=fluentd_server[0],
            port=fluentd_port,
            queue_circular=True,
        )

    def emit(self, result: pd.DataFrame):
        records = json.loads(result.to_json(orient="records"))
        for data in records:
            self._sender.emit(label=None, data=data)

    def stop(self):
        self._sender.close()
