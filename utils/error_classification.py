"""Shared permanent-vs-transient-vs-fatal error classification — one rule
set, used by core/kafka_worker_entrypoint.py's DLQ, core/pipeline.py's bulk
failure ledger, repository/neo4j_repository.py's per-call retry,
services/embedding_service.py / services/inhouse_embedding_client.py's embed
retry loops, and connectors/elasticsearch_client.py's per-page retry, so the
same exception type gets the same verdict everywhere instead of
independently-drifting copies. (services/extraction_service.py's structured
LLM call and services/llm_client.py's confirm_with_retry /
call_with_retry use classify_llm_retry_error() below instead — same rule
set, except a bad generation is transient there.)

fatal: a global, config-level problem that would fail identically for every
future message too, not just this one — bad Neo4j credentials (401/403),
wrong database name, a bad Azure OpenAI API key/endpoint, bad Elasticsearch
credentials, etc. The kafka worker crashes outright on this rather than
draining the entire topic into the DLQ one message at a time, which would
turn a one-line config fix into a mass data-loss event;
services/extraction_service.py re-raises it rather than quietly marking one
chunk failed=True and retrying every future one the same losing way.

permanent: retrying the exact same input would fail identically forever, but
only for THIS input, not every future one — malformed JSON/structure
(JSONDecodeError), a missing required field (KeyError), a response that fails
schema validation (pydantic's ValidationError — e.g. a structurally-off model response) or
comes back with no choices at all (IndexError), a bad Neo4j request (400) or
missing resource (404), an S3 claim-check pointer that will never resolve
(ClientError 403/404 — access denied or the object genuinely doesn't exist), an
Azure OpenAI content-filter/bad-request rejection (400), a bad Elasticsearch
query (400) or missing index (404) — that input triggered it and will trigger
it again no matter how many times it's retried.

transient: everything else — network blips, Neo4j 5xx/timeouts, Azure
throttling that escaped a service's own retry, S3 5xx/throttling (also
ClientError, but not 403/404), an Azure OpenAI 5xx/connection/timeout error,
an Elasticsearch 5xx/connection/timeout error. Worth a bounded number of
redeliveries (Kafka), a --retry-failed rerun (bulk), or a short in-call retry
(extraction, Neo4j, Elasticsearch) — the underlying blip might have cleared
by then.

Deliberately narrow on bare ValueError: only json.JSONDecodeError (a
ValueError subclass, listed explicitly for clarity) counts as permanent — an
unrelated ValueError from some other, possibly-retryable code path must not
be silently written off as unfixable without ever retrying.

repository.run_registry.NoCompletedRunError is also permanent — nothing to
write into yet, and retrying won't change that.

It also disagrees on a 404 from the chat gateway: permanent for every other
caller (a missing Neo4j database/ES index/S3 key is not coming back), but
transient for an LLM call, because hosted gateways return it intermittently
for a model that exists — see classify_llm_retry_error's own comment.

classify_llm_retry_error() is the verdict for bounded in-call LLM retries —
services/llm_client.py's confirm_with_retry / call_with_retry (merge
confirm, cluster resolution) and services/extraction_service.py's
_structured_call (base extraction + gleaning, both Azure and Ministral via
llm_client.complete()). It deliberately disagrees with
classify_error() on bad model output: invalid JSON, schema validation
failure, an empty choices list, or a missing expected key in a parsed reply
are treated as transient there — a regenerate can succeed — whereas
classify_error() marks those permanent for Kafka/DLQ and other paths where
re-processing the exact same bytes would not help.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from repository.run_registry import NoCompletedRunError

_FATAL_HTTP_STATUSES = {401, 403}
_PERMANENT_HTTP_STATUSES = {400, 404}


class LlmTruncationError(ValueError):
    """The model ran out of output room (finish_reason='length') — the chunk's
    extraction genuinely needs more tokens than the context window has left.

    Defined here (not in services/llm_client.py, which raises it) so this
    module can classify it without a circular import. Deliberately PERMANENT
    for the in-call retry loop: generation is deterministic at temperature 0,
    so a retry produces the identical truncation — a real run (x9j2, chunk
    bf9e6b8a) burned 3 identical ~2-minute attempts proving this. Recovery is
    structural, not a retry: services/extraction_service.py repair-parses the
    partial output carried on this exception and asks one follow-up call for
    only what was missed.

    partial_content: the truncated raw completion text, when the raiser had
    it — everything the model generated before hitting the ceiling. The
    repair path salvages the complete objects out of it rather than paying
    for their re-generation."""

    def __init__(self, message: str, partial_content: "str | None" = None) -> None:
        super().__init__(message)
        self.partial_content = partial_content


def classify_error(exc: Exception) -> str:
    if isinstance(exc, NoCompletedRunError):
        return "permanent"
    try:
        from requests.exceptions import HTTPError
    except ImportError:
        HTTPError = None  # noqa: N806 — requests isn't a dependency of every caller
    if HTTPError is not None and isinstance(exc, HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status in _FATAL_HTTP_STATUSES:
            return "fatal"
        if status in _PERMANENT_HTTP_STATUSES:
            return "permanent"
        return "transient"
    # ValidationError IS a ValueError subclass, so it must be checked ahead of/
    # separately from the deliberately-narrow bare-ValueError default below —
    # a model response that fails schema validation will fail the same way on
    # every retry, unlike an unrelated, possibly-transient ValueError.
    if isinstance(exc, (json.JSONDecodeError, KeyError, ValidationError, IndexError)):
        return "permanent"
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        ClientError = None  # noqa: N806 — botocore isn't a dependency of every caller
    if ClientError is not None and isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = exc.response.get("Error", {}).get("Code", "")
        if status in (403, 404) or error_code in ("403", "404", "AccessDenied", "NoSuchKey", "NoSuchBucket"):
            return "permanent"
        return "transient"
    try:
        # RateLimitError (429) is deliberately NOT special-cased here — every
        # caller that talks to Azure OpenAI already has its own dedicated
        # rate-limit handling (adaptive backoff tied into AdaptiveThrottle),
        # so it never reaches this generic classifier in practice.
        from openai import APIConnectionError, APIStatusError
    except ImportError:
        APIStatusError = APIConnectionError = None  # noqa: N806 — openai isn't a dependency of every caller
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        status = exc.status_code
        if status in _FATAL_HTTP_STATUSES:
            return "fatal"
        if status in _PERMANENT_HTTP_STATUSES:
            return "permanent"
        return "transient"
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return "transient"  # covers APITimeoutError too (subclasses APIConnectionError)
    try:
        from elastic_transport import TransportError
        from elasticsearch import ApiError
    except ImportError:
        ApiError = TransportError = None  # noqa: N806 — elasticsearch isn't a dependency of every caller
    if ApiError is not None and isinstance(exc, ApiError):
        status = exc.meta.status if exc.meta is not None else None
        if status in _FATAL_HTTP_STATUSES:
            return "fatal"
        if status in _PERMANENT_HTTP_STATUSES:
            return "permanent"
        return "transient"
    if TransportError is not None and isinstance(exc, TransportError):
        return "transient"  # connection errors/timeouts — no status code, never permanent/fatal
    try:
        # qdrant_client raises UnexpectedResponse (not requests.HTTPError) for
        # missing collections etc. — without this, a permanent 404 is retried as
        # transient by dual_write_guard (~3 backoffs per entity).
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:
        UnexpectedResponse = None  # noqa: N806
    if UnexpectedResponse is not None and isinstance(exc, UnexpectedResponse):
        status = exc.status_code
        if status in _FATAL_HTTP_STATUSES:
            return "fatal"
        if status in _PERMANENT_HTTP_STATUSES:
            return "permanent"
        return "transient"
    return "transient"


def classify_llm_retry_error(exc: Exception) -> str:
    """Bounded in-call LLM retry verdict — see module docstring for why this
    differs from classify_error() on bad generations.

    Truncation (finish_reason='length') is PERMANENT here, unlike other bad
    generations: a regenerate at temperature 0 is byte-identical, so it hits
    the same ceiling every time. Matched before the bad-generation branch —
    LlmTruncationError is a ValueError, and openai's own
    LengthFinishReasonError is checked lazily like every other openai type."""
    if isinstance(exc, LlmTruncationError):
        return "permanent"
    try:
        from openai import LengthFinishReasonError
    except ImportError:
        LengthFinishReasonError = None  # noqa: N806 — openai isn't a dependency of every caller
    if LengthFinishReasonError is not None and isinstance(exc, LengthFinishReasonError):
        return "permanent"
    if isinstance(exc, (json.JSONDecodeError, ValidationError, IndexError, KeyError)):
        return "transient"
    # A 404 from the CHAT GATEWAY is transient here, unlike everywhere else
    # (classify_error keeps it permanent for Neo4j/ES/S3/Kafka, where 404
    # genuinely means "this resource does not exist"). Hosted OpenAI-compatible
    # gateways return 404 intermittently for a model that demonstrably exists —
    # NVIDIA NIM answers "Function '<uuid>': Not found for account" for perhaps
    # 1 call in 10 on a model that the very next identical call serves fine.
    # Failing closed on that cost real data: a bulk run silently wrote an EMPTY
    # extraction for a chunk (0 entities, cached as "genuinely nothing here")
    # after a single attempt. Whether the model name is actually wrong is
    # already settled at startup by llm_backend's reachability probe; a genuinely
    # bad name still fails loudly here, just after the bounded retries instead of
    # on the first attempt.
    try:
        from openai import APIStatusError
    except ImportError:
        APIStatusError = None  # noqa: N806 — openai isn't a dependency of every caller
    if APIStatusError is not None and isinstance(exc, APIStatusError) and exc.status_code == 404:
        return "transient"
    return classify_error(exc)
