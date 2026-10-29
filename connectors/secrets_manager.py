"""AWS Secrets Manager client — the bootstrap source for local_vault mode's
Vault AppRole credentials (connectors/vault_manager.py's LocalVaultManager).

boto3 is already a hard dependency of this repo (used elsewhere), so this
import is not guarded — unlike ai-graph-retrieval, which treats it as an
optional platform extra.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.client import Config

logger = logging.getLogger("graphrag.connectors.secrets_manager")

_STS_PROBE_CONFIG = Config(
    connect_timeout=5, read_timeout=5, retries={"max_attempts": 2, "mode": "standard"}
)


def _session_is_active(session: boto3.Session, region: Optional[str]) -> bool:
    try:
        if session.get_credentials() is None:
            return False
        session.client("sts", region_name=region, config=_STS_PROBE_CONFIG).get_caller_identity()
        return True
    except Exception as exc:
        logger.warning("AWS credential source not active: %s", exc.__class__.__name__)
        return False


def _build_active_session(region: Optional[str], profile: Optional[str]) -> boto3.Session:
    have_static = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))

    candidates: list[tuple[str, boto3.Session]] = []
    if have_static:
        candidates.append(("static env credentials", boto3.Session(region_name=region)))
    if profile:
        candidates.append((f"profile {profile!r}", boto3.Session(profile_name=profile, region_name=region)))
    if not candidates:
        candidates.append(("default provider chain", boto3.Session(region_name=region)))

    for label, session in candidates:
        if _session_is_active(session, region):
            logger.info("AWS auth: using %s", label)
            return session

    logger.error("AWS auth: no active credential source (%s) — proceeding with %s so its "
                 "underlying error surfaces", ", ".join(l for l, _ in candidates), candidates[0][0])
    return candidates[0][1]


class SecretsManager:
    """Thin wrapper over boto3 Secrets Manager. get_secret returns a dict
    parsed from the secret's JSON SecretString (or base64 SecretBinary)."""

    def __init__(self) -> None:
        # Prefer AWS_DEFAULT_REGION (Helm/configMap); fall back to AWS_REGION.
        # A missing region raises NoRegionError deep inside botocore — surface it
        # here only after both env names are checked.
        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
        profile = os.getenv("AWS_PROFILE")
        session = _build_active_session(region, profile)
        self._client = session.client(
            "secretsmanager",
            config=Config(signature_version="v4", retries={"max_attempts": 10, "mode": "standard"}),
        )

    def get_secret(self, secret_name: str) -> dict[str, Any]:
        try:
            resp = self._client.get_secret_value(SecretId=secret_name)
            raw = resp.get("SecretString") or base64.b64decode(resp["SecretBinary"])
            return json.loads(raw)
        except Exception:
            # Never log the secret's contents; the secret NAME is safe to log.
            logger.error("AWS Secrets Manager fetch failed for %s", secret_name)
            raise
