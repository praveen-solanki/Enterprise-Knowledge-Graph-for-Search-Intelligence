"""
Turns text into embedding vectors. ONE door — embed_batch — for every vector
this pipeline computes, entity or chunk: the run's embedding space (backend,
model, dimension — services/run_context.RunConfig, mirrored from
config/settings.py's embedding-space fields) decides where the request goes,
so no caller can ever pick a model or dimension of its own and land outside
the run's space. Nothing else in this file — no clustering, no LLM merge
decisions.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from openai import AzureOpenAI, BadRequestError, RateLimitError

from config.settings import settings
from utils.error_classification import classify_error
from services.inhouse_embedding_client import embed_batch as inhouse_embed_batch
from services.llm_backend import effective_run_config
from services.run_context import RunConfig
from services.run_metrics import record

logger = logging.getLogger("graphrag.services.embedding")

# Azure's embeddings endpoint caps inputs per request at ~2048; sub-batching
# here means every caller gets this for free instead of hand-batching itself.
_EMBED_BATCH_SIZE = settings.embedding_azure_batch_size
_EMBED_MAX_RETRIES = settings.embedding_azure_max_retries
_EMBED_BASE_BACKOFF = settings.embedding_azure_base_backoff_seconds
_EMBED_MAX_BACKOFF = settings.embedding_azure_max_backoff_seconds

# Shared cooldown so concurrent embed workers don't stampede after a 429.
_rate_limit_lock = threading.Lock()
_rate_limit_until = 0.0
_RETRY_AFTER_RE = re.compile(r"retry after\s+(\d+)\s+seconds?", re.IGNORECASE)


def _parse_retry_after_seconds(exc: BaseException, default: float) -> float:
    """Prefer Azure's 'retry after N seconds' hint over blind exponential backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + random.uniform(0.1, 0.5)
    return default


def _wait_out_rate_limit_cooldown() -> None:
    with _rate_limit_lock:
        delay = _rate_limit_until - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _note_rate_limit_cooldown(seconds: float) -> None:
    global _rate_limit_until
    with _rate_limit_lock:
        _rate_limit_until = max(_rate_limit_until, time.monotonic() + seconds)


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    """Cheap upper-bound token estimate for packing (not a real tokenizer)."""
    return max(1, int(len(text) / chars_per_token) + 1)


def _is_azure_request_too_large(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "maximum request size" in msg or "too large" in msg


def _split_into_batches(
    texts: list[str],
    *,
    max_items: int,
    max_tokens: Optional[int] = None,
    chars_per_token: float = 3.0,
) -> list[list[str]]:
    """Pack texts into request batches by item count and optional token budget.

    When max_tokens is set (Azure), a batch is closed before adding the next
    text would exceed the budget. A single oversized text still goes alone —
    Azure's per-input limit is a separate failure mode.
    """
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_tokens is None:
        return [texts[i:i + max_items] for i in range(0, len(texts), max_items)]

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text, chars_per_token)
        would_exceed = (
            current
            and (len(current) >= max_items or current_tokens + tokens > max_tokens)
        )
        if would_exceed:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


class EmbeddingService:
    def __init__(self) -> None:
        # The Azure client is built lazily on the first Azure-bound batch, not
        # here: a run whose embedding space is "inhouse" never touches Azure for
        # embeddings and must not be required to carry Azure embedding
        # credentials just to construct this service.
        self._azure_client: Optional[AzureOpenAI] = None

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embeds any number of texts in the running run's embedding space —
        the ONE embedding entry point for the whole pipeline.

        The space (backend, model, dimension) is the run's recorded RunConfig
        (identical to settings on the bulk path; the bulk run's own recorded
        values on an incremental event), so every vector — entity or chunk —
        lands in the same space as the run that owns it. An unreachable backend
        fails the work after retries; it never falls back to the other backend,
        which would silently mix two models' vectors in one collection.

        Blank/whitespace-only strings are replaced with a single space — Azure
        rejects a bare empty string outright, which would otherwise fail the
        whole batch over one bad entry; the in-house gateway gets the same
        treatment for identical semantics.

        Azure batches are capped by both input count and a total-token budget
        (Azure rejects ~300k tokens/request). In-house batches stay count-only.

        Retries transient failures (rate limits, 5xx, connection resets) with
        exponential backoff; fatal errors propagate immediately.

        For corpus-scale clustering prefer embed_batch_matrix() — that path
        packs float32 and avoids a giant nested-Python-float RSS spike."""
        if not texts:
            return []
        cfg = effective_run_config()
        batches = self._split_texts(texts, cfg)
        if len(batches) == 1:
            return self._embed_one_batch(batches[0], cfg)
        concurrency = (settings.embedding_azure_concurrency if cfg.embedding_backend == "azure"
                      else settings.embedding_inhouse_concurrency)
        workers = min(concurrency, len(batches))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed") as pool:
            return [vector
                    for chunk in pool.map(lambda batch: self._embed_one_batch(batch, cfg), batches)
                    for vector in chunk]

    def embed_batch_matrix(self, texts: list[str]) -> np.ndarray:
        """Like embed_batch, but returns one float32 matrix (n × dim).

        Each sub-batch is cast to float32 immediately and stacked, so peak RSS
        stays near the packed matrix size instead of list[list[float]].
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        cfg = effective_run_config()
        batches = self._split_texts(texts, cfg)
        if len(batches) == 1:
            rows = self._embed_one_batch(batches[0], cfg)
            return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
        concurrency = (settings.embedding_azure_concurrency if cfg.embedding_backend == "azure"
                      else settings.embedding_inhouse_concurrency)
        workers = min(concurrency, len(batches))
        chunks: list[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed") as pool:
            for rows in pool.map(lambda batch: self._embed_one_batch(batch, cfg), batches):
                chunks.append(np.ascontiguousarray(np.asarray(rows, dtype=np.float32)))
                del rows
        return np.vstack(chunks)

    @staticmethod
    def _split_texts(texts: list[str], cfg: RunConfig) -> list[list[str]]:
        # Each backend has its own per-request input cap: Azure ~2048, the
        # in-house gateway 512 (it 502s above that) — same split as the two
        # backends' concurrency below. Azure also has a total-token cap.
        if cfg.embedding_backend == "inhouse":
            return _split_into_batches(
                texts, max_items=settings.embedding_inhouse_batch_size,
            )
        return _split_into_batches(
            texts,
            max_items=_EMBED_BATCH_SIZE,
            max_tokens=settings.embedding_azure_max_tokens_per_request,
            chars_per_token=settings.embedding_azure_chars_per_token_estimate,
        )

    def _embed_one_batch(self, batch: list[str], cfg: RunConfig) -> list[list[float]]:
        """Routes one sanitized sub-batch to the space's backend."""
        sanitized = self._sanitize(batch)
        if cfg.embedding_backend == "inhouse":
            record("embedding_calls")
            return inhouse_embed_batch(sanitized, model=cfg.embedding_model,
                                       dimensions=cfg.embedding_dimension)
        return self._azure_embed_batch(sanitized, dimensions=cfg.embedding_dimension)

    @staticmethod
    def _sanitize(batch: list[str]) -> list[str]:
        sanitized = [t if t and t.strip() else " " for t in batch]
        empty_count = sum(1 for t in batch if not t or not t.strip())
        if empty_count:
            logger.warning("Replaced %d empty/whitespace string(s) before embedding request", empty_count)
        return sanitized

    def _get_azure_client(self) -> AzureOpenAI:
        if self._azure_client is None:
            # A dedicated embedding-only Azure resource, when Vault's llm secret
            # provisions one (AZURE_OPENAI_EMBEDD_ENDPOINT/_API_KEY) — falls back
            # to the shared chat/extraction credentials otherwise, so this works
            # unchanged in an environment that only sets the shared pair. Endpoint
            # and key are an all-or-nothing pair: a half-configured dedicated
            # resource (only one of the two set) would otherwise combine a
            # dedicated endpoint with a shared key (or vice versa), which Azure
            # rejects since credentials are scoped to one specific resource.
            if settings.azure_openai_embedding_endpoint and settings.azure_openai_embedding_api_key:
                endpoint = settings.azure_openai_embedding_endpoint
                api_key = settings.azure_openai_embedding_api_key
            else:
                endpoint = settings.azure_openai_endpoint
                api_key = settings.azure_openai_api_key
            if not endpoint or not api_key:
                raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")
            self._azure_client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=settings.azure_openai_api_version,
                # Without this, a half-closed Azure LB connection can hang forever
                # (CLOSE-WAIT) with no log line after "Embedding N text(s)".
                timeout=settings.azure_openai_request_timeout,
            )
        return self._azure_client

    def _azure_embed_batch(self, sanitized: list[str], *, dimensions: int) -> list[list[float]]:
        delay = 0.0
        for attempt in range(_EMBED_MAX_RETRIES):
            _wait_out_rate_limit_cooldown()
            if delay > 0:
                time.sleep(delay)
            try:
                logger.debug("Embedding %d text(s) via %s", len(sanitized), settings.azure_openai_embedding_deployment)
                record("embedding_calls")
                response = self._get_azure_client().embeddings.create(
                    model=settings.azure_openai_embedding_deployment,
                    input=sanitized,
                    dimensions=dimensions,
                )
                return [d.embedding for d in response.data]
            except BadRequestError as exc:
                # Token-estimate packing can still under-shoot the real tokenizer.
                # Halve and retry rather than failing the whole Neo4j write batch.
                if len(sanitized) > 1 and _is_azure_request_too_large(exc):
                    mid = len(sanitized) // 2
                    logger.warning(
                        "Azure embedding request too large (%d texts); splitting into %d + %d",
                        len(sanitized), mid, len(sanitized) - mid,
                    )
                    return (
                        self._azure_embed_batch(sanitized[:mid], dimensions=dimensions)
                        + self._azure_embed_batch(sanitized[mid:], dimensions=dimensions)
                    )
                raise
            except RateLimitError as exc:
                if attempt == _EMBED_MAX_RETRIES - 1:
                    raise
                delay = min(
                    _parse_retry_after_seconds(
                        exc, _EMBED_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1),
                    ),
                    _EMBED_MAX_BACKOFF,
                )
                _note_rate_limit_cooldown(delay)
                logger.warning("Embedding rate limited (attempt %d/%d), retrying in %.1fs: %s",
                               attempt + 1, _EMBED_MAX_RETRIES, delay, exc)
            except Exception as exc:  # noqa: BLE001 — classified below, not swallowed
                if classify_error(exc) != "transient" or attempt == _EMBED_MAX_RETRIES - 1:
                    raise
                delay = min(_EMBED_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1), _EMBED_MAX_BACKOFF)
                logger.warning("Embedding call failed (transient, attempt %d/%d), retrying in %.1fs: %s",
                               attempt + 1, _EMBED_MAX_RETRIES, delay, exc)
        raise RuntimeError("unreachable")  # loop always returns or raises


_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """One shared instance, reused everywhere — not a fresh client per call."""
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
