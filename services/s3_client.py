"""
S3 claim-check download for the incremental Kafka payload. The orchestrator
stages large event payloads to S3 and hands the consumer a pointer instead
of embedding them inline on Kafka — see
GRAPH_INCREMENTAL_SYNC_ORCHESTRATOR_DESIGN.md §3-4. No retry/backoff here —
a failed download propagates to the caller, which leaves the Kafka message
uncommitted for redelivery, same as any other processing error.

A single module-level client, built once and reused for the life of the
process (not a fresh one per call, which would mean a brand-new TCP
connection and TLS handshake per Kafka message), with explicit connect/read
timeouts so a stalled call fails fast into the DLQ/retry path instead of
hanging the worker on one message indefinitely.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import boto3
from botocore.config import Config

from config.settings import settings

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = boto3.client(
                    "s3",
                    region_name=settings.aws_region,
                    config=Config(
                        connect_timeout=settings.s3_connect_timeout_seconds,
                        read_timeout=settings.s3_read_timeout_seconds,
                    ),
                )
    return _client


def download_json(bucket: str, key: str) -> dict[str, Any]:
    client = _get_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())
