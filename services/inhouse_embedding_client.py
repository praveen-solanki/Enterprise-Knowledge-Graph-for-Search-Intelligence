"""Client for the internally-hosted embeddings gateway — the transport half of
the "inhouse" embedding backend.

Which model and dimension to request is the CALLER's decision
(services/embedding_service.py passes them from the run's RunConfig / settings
embedding-space fields); this module never picks a model itself. Response shape
verified live against the real dev endpoint (snowflake-arctic-embed-l-v2.0
returns HTTP 200 with real vectors):

    {"status": "success", "model": "...", "dimension": 256,
     "results": [{"id": "chunk-1", "embedding": [...]}],
     "additional_info": ..., "metadata": ...}
"""
from __future__ import annotations

import logging
import random
import time

import requests

from config.settings import settings
from utils.error_classification import classify_error

logger = logging.getLogger("graphrag.services.inhouse_embedding_client")

_MAX_RETRIES = settings.embedding_inhouse_max_retries
_BASE_BACKOFF = settings.embedding_inhouse_base_backoff_seconds
_MAX_BACKOFF = settings.embedding_inhouse_max_backoff_seconds
_REQUEST_TIMEOUT = settings.embedding_inhouse_request_timeout_seconds


def _is_openai_embeddings_url(url: str) -> bool:
    """vLLM and other OpenAI-compatible servers expose POST /v1/embeddings.
    The older in-house gateway uses its own /embed payload instead."""
    return url.split("?", 1)[0].rstrip("/").endswith("/embeddings")


def _vectors_from_response(body: dict, ids: list[str]) -> list[list[float]]:
    if "data" in body:
        rows = sorted(body["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]
    by_id = {row["id"]: row["embedding"] for row in body["results"]}
    return [by_id[i] for i in ids]


def embed_batch(texts: list[str], *, model: str, dimensions: int) -> list[list[float]]:
    """POSTs one batch to the in-house embeddings gateway, returns vectors in
    the same order as `texts` — matched back by the `id` each input was given,
    not by response array position, in case the service ever reorders results.

    Retry/backoff mirrors EmbeddingService._azure_embed_batch, so callers see
    identical failure semantics regardless of which backend is active. An
    unreachable gateway therefore FAILS the work after the retries — never
    falls back to another model, which would silently mix embedding spaces."""
    if not texts:
        return []
    ids = [str(i) for i in range(len(texts))]
    cleaned = [t if t and t.strip() else " " for t in texts]
    url = settings.in_house_embeddings_url
    if _is_openai_embeddings_url(url):
        # OpenAI / vLLM: {"input": [...], "model", "dimensions"} -> data[].embedding.
        # NIM's embed models take input_type and a fixed dimension, and reject
        # the OpenAI "dimensions" field. settings.inference_provider is the
        # explicit switch (config/settings.py) — not URL sniffing.
        if settings.inference_provider == "nvidia":
            payload = {"input": cleaned, "model": model, "input_type": "passage",
                       "encoding_format": "float", "truncate": "END"}
        else:
            payload = {"input": cleaned, "model": model, "dimensions": dimensions}
        headers = {"Content-Type": "application/json"}
        if settings.ministral_api_key:
            headers["Authorization"] = f"Bearer {settings.ministral_api_key}"
    else:
        payload = {
            "inputs": [{"id": i, "text": t} for i, t in zip(ids, cleaned)],
            "include_metadata": True,
            "model": model,
            "dim": dimensions,
            "quantize_int8": settings.in_house_embeddings_quantize,
        }
        headers = {
            "Content-Type": "application/json",
            f"{settings.in_house_embeddings_header_prefix}-service-code": settings.in_house_embeddings_service_code,
            f"{settings.in_house_embeddings_header_prefix}-embedding-priority": "async",
        }

    delay = 0.0
    for attempt in range(_MAX_RETRIES):
        if delay > 0:
            time.sleep(delay)
        try:
            logger.debug("Embedding %d text(s) via in-house service (%s @ %dd)",
                         len(texts), model, dimensions)
            response = requests.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            return _vectors_from_response(response.json(), ids)
        except Exception as exc:  # noqa: BLE001 — classified below, not swallowed
            if classify_error(exc) != "transient" or attempt == _MAX_RETRIES - 1:
                raise
            delay = min(_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1), _MAX_BACKOFF)
            logger.warning("In-house embedding call failed (transient, attempt %d/%d), "
                           "retrying in %.1fs: %s", attempt + 1, _MAX_RETRIES, delay, exc)
    raise RuntimeError("unreachable")  # loop always returns or raises
