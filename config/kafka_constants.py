"""Kafka topic/consumer-group naming templates — single source of truth for
how a per-env name is derived, instead of a fully-resolved literal string
duplicated across all 13 helm-values files (real copy-paste-drift risk: one
env file's Kafka name only has to diverge once for a worker to silently
consume the wrong topic). Helm only needs to set DEPLOY_ENV (the env slug
matching its own filename, e.g. "uat-eu"); config/settings.py derives the
concrete topic/group names from these templates.

Casing is deliberately inconsistent across the three names below — Title for
the main topic, UPPER for the DLQ, lower for the consumer group — because
that's what config/settings.py's own prior hardcoded defaults already used
(AI_Graph_Extract_Req_Dev / AI_Graph_Extract_Req_DLQ_DEV /
graphrag-ingestion-service-group-dev). Preserved exactly as-is rather than
"fixed", since these may already be real, provisioned Kafka topics — CONFIRM
with whoever owns the Kafka cluster before ever changing the casing.
"""
from __future__ import annotations

KAFKA_TOPIC_GRAPH_EXTRACT_TEMPLATE = "AI_Graph_Extract_Req_{env}"
KAFKA_TOPIC_GRAPH_EXTRACT_DLQ_TEMPLATE = "AI_Graph_Extract_Req_DLQ_{env}"
KAFKA_CONSUMER_GROUP_TEMPLATE = "graphrag-ingestion-service-group-{env}"


def title_env(deploy_env: str) -> str:
    """"uat-eu" -> "Uat_Eu" (matches the pre-existing "Dev"/"Qa"/"Prod" style)."""
    return "_".join(part.capitalize() for part in deploy_env.split("-"))


def upper_env(deploy_env: str) -> str:
    """"uat-eu" -> "UAT_EU" (matches the pre-existing "DEV" style)."""
    return deploy_env.upper().replace("-", "_")


def lower_env(deploy_env: str) -> str:
    """"uat-eu" -> "uat-eu" (matches the pre-existing "dev" style)."""
    return deploy_env.lower()
